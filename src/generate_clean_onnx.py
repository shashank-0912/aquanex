import torch
import torch.nn as nn
import os

class CausalANCModel(nn.Module):
    def __init__(self, input_dim=257, hidden_dim=128, num_layers=2):
        super().__init__()
        # Matches fc_in.weight [128, 257], fc_in.bias [128]
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        
        # Matches gru.weight_ih_l0 [384, 128], etc.
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Matches fc_out.weight [257, 128], fc_out.bias [257]
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, h):
        # x: [batch, 1, 257]
        x_proj = torch.relu(self.fc_in(x))
        out, h_next = self.gru(x_proj, h)
        mask = torch.sigmoid(self.fc_out(out))
        return mask, h_next

def export():
    model = CausalANCModel(input_dim=257, hidden_dim=128, num_layers=2)
    
    weights_path = "model_tactical_v2.pth"
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[+] Successfully loaded trained weights from: {weights_path}")
    else:
        print(f"[!] Error: {weights_path} not found.")
        return

    model.eval()

    # 1 audio frame: [batch_size=1, time_steps=1, freq_bins=257]
    dummy_x = torch.randn(1, 1, 257, dtype=torch.float32)
    # 2-layer GRU recurrent state: [num_layers=2, batch_size=1, hidden_dim=128]
    dummy_h = torch.zeros(2, 1, 128, dtype=torch.float32)

    output_onnx = "aquanex_causal_gru.onnx"
    
    if os.path.exists(output_onnx):
        os.remove(output_onnx)

    torch.onnx.export(
        model,
        (dummy_x, dummy_h),
        output_onnx,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_spectrum', 'hidden_state_in'],
        output_names=['gain_mask', 'hidden_state_out'],
        dynamic_axes={
            'input_spectrum': {0: 'batch_size'},
            'hidden_state_in': {1: 'batch_size'},
            'gain_mask': {0: 'batch_size'},
            'hidden_state_out': {1: 'batch_size'}
        }
    )
    
    size_kb = os.path.getsize(output_onnx) / 1024
    print(f"[+] Model exported cleanly: {output_onnx} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    export()