import torch
from model import MultimodalDrivingModel

def export_to_onnx():
    print("Loading PyTorch model...")
    model = MultimodalDrivingModel()
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    
    # Create dummy inputs with the exact same shapes used during training
    # Batch size of 1 is required for inference
    dummy_vision = torch.randn(1, 4, 192, 320)
    dummy_bev = torch.randn(1, 1, 128, 128)
    dummy_meta = torch.randn(1, 7)
    
    onnx_file_path = "best_model.onnx"
    
    print(f"Exporting model to {onnx_file_path}...")
    torch.onnx.export(
        model, 
        (dummy_vision, dummy_bev, dummy_meta), 
        onnx_file_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['vision', 'bev', 'meta'],
        output_names=['actions'],
        dynamic_axes={
            'vision': {0: 'batch_size'},
            'bev': {0: 'batch_size'},
            'meta': {0: 'batch_size'},
            'actions': {0: 'batch_size'}
        }
    )
    
    print("ONNX Export Successful! The model is now ready for ultra-fast, low-VRAM inference.")

if __name__ == '__main__':
    export_to_onnx()
