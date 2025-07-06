import torch
import pickle


ckpt = torch.load(r"C:\Users\Kirit\.cache\diffusion_models_converted\ema_diffusion_cifar10_model\model-790000.ckpt", map_location="cpu")
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

converted_state = {}
for k, v in state_dict.items():
    if isinstance(v, torch.Tensor):
        converted_state[k] = v.cpu().numpy()
    else:
        converted_state[k] = v  # 通常不会进这里

with open(r"C:\Users\Kirit\.cache\diffusion_models_converted\ema_diffusion_cifar10_model\model-790000.pkl", "wb") as f:
    pickle.dump(converted_state, f)

print("Converted model saved to model-2388000.pkl")