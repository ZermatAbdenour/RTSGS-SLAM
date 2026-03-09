from gsplat import rendering, spherical_harmonics
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 8
with torch.no_grad():
    rendering.rasterization(
        means=torch.randn(N, 3, device=device),
        quats=F.normalize(torch.randn(N, 4, device=device), dim=-1),
        scales=torch.full((N, 3), 0.01, device=device),
        opacities=torch.ones(N, device=device),
        colors=torch.ones(N, 3, device=device),
        viewmats=torch.eye(4, device=device).unsqueeze(0),
        Ks=torch.tensor([[100.,0,320],[0,100.,240],[0,0,1]], device=device).unsqueeze(0),
        width=64, height=64,
    )
    # Also warm the ED render mode used in process_single_keyframe
    rendering.rasterization(
        means=torch.randn(N, 3, device=device),
        quats=F.normalize(torch.randn(N, 4, device=device), dim=-1),
        scales=torch.full((N, 3), 0.01, device=device),
        opacities=torch.ones(N, device=device),
        colors=torch.ones(N, 3, device=device),
        viewmats=torch.eye(4, device=device).unsqueeze(0),
        Ks=torch.tensor([[100.,0,320],[0,100.,240],[0,0,1]], device=device).unsqueeze(0),
        width=64, height=64,
        render_mode="ED",
    )
torch.cuda.synchronize()