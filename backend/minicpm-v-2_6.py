from transformers import AutoTokenizer, AutoModel

from vision_qna import *
from PIL import Image
from decord import VideoReader, cpu

# openbmb/MiniCPM-V-2_6
# openbmb/MiniCPM-V-2_6-int4

MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number

async def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames


class VisionQnA(VisionQnABase):
    format: str = 'internal'
    model_name: str = "minicpm-v-2_6"
    vision_layers: List[str] = ["resampler", "vpm"]
    
    def __init__(self, model_id: str, device: str, device_map: str = 'auto', extra_params = {}, format = None):
        super().__init__(model_id, device, device_map, extra_params, format)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=self.params.get('trust_remote_code', False))
        self.model = AutoModel.from_pretrained(**self.params).eval()

        # bitsandbytes already moves the model to the device, so we don't need to do it again.
        if '-int4' not in model_id:
            if not (extra_params.get('load_in_4bit', False) or extra_params.get('load_in_8bit', False)):
                self.model = self.model.to(dtype=self.params['torch_dtype'], device=self.device)
    
        self.loaded_banner()
    
    async def stream_chat_with_images(self, request: ImageChatRequest) -> AsyncGenerator[str, None]:
        msgs = []
        system_prompt = None

        params = self.get_generation_params(request)

        for m in request.messages:
            image = None
            content = []
            for c in m.content:
                if m.role == 'user':
                    if c.type == 'image_url':
#                        # Video not working yet
#                        if '.mp4' in c.image_url.url:
#                            params["use_image_id"] = False
#                            params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution >  448*448
#                            video_path = await url_to_file(c.image_url.url)
#                            image = await encode_video(video_path)
                        content.extend([await url_to_image(c.image_url.url)])

            for c in m.content:
                if c.type == 'text':
                    if m.role == 'system':
                        system_prompt = c.text
                    else:
                        content.extend([c.text])
                        msgs.extend([{ 'role': m.role, 'content': content }])

        default_params = dict(
            do_sample=True,
            top_p=0.8,
            temperature=0.7,
        )

        params = self.get_generation_params(request, default_params=default_params)

        answer = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            system_prompt=system_prompt,
            stream=True,
            **params,
        )

        for new_text in answer:
            yield new_text

