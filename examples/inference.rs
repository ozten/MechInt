use burn::{
    backend::{wgpu::WgpuDevice, Autodiff, Wgpu},
    module::AutodiffModule,
    tensor::{Int, Tensor},
};

use grokking::checkpoint;

type Backend = Wgpu;
type MyAutodiffBackend = Autodiff<Backend>;

fn main() {
    println!("🔍 Loading trained model from checkpoint...");
    println!();

    let device = WgpuDevice::default();

    // Load the final trained model
    let model = checkpoint::load_checkpoint::<MyAutodiffBackend>("checkpoints/model_final", &device)
        .expect("Failed to load model checkpoint");

    println!();
    println!("✅ Model loaded successfully!");
    println!();

    // Test some examples
    println!("🧪 Testing modular addition (mod 97):");
    println!("{}", "=".repeat(60));
    println!();

    let test_cases = vec![
        (5, 10),      // 5 + 10 = 15
        (50, 50),     // 50 + 50 = 100 mod 97 = 3
        (48, 49),     // 48 + 49 = 97 mod 97 = 0
        (96, 96),     // 96 + 96 = 192 mod 97 = 95
        (12, 34),     // 12 + 34 = 46
        (60, 60),     // 60 + 60 = 120 mod 97 = 23
        (88, 88),     // 88 + 88 = 176 mod 97 = 79
        (1, 1),       // 1 + 1 = 2
        (0, 97),      // 0 + 97 = 97 mod 97 = 0
        (45, 52),     // 45 + 52 = 97 mod 97 = 0
    ];

    for (a, b) in test_cases {
        let expected = (a + b) % 97;
        let input_vec = vec![a as i32, b as i32, 97]; // 97 is the '=' token

        let input = Tensor::<MyAutodiffBackend, 1, Int>::from_ints(input_vec.as_slice(), &device)
            .reshape([1, 3]);

        // Run inference (no gradients needed)
        let logits = model.clone().valid().forward(input.inner());
        let prediction = logits.argmax(1).squeeze::<1>();
        let predicted: i32 = prediction.into_data().to_vec().unwrap()[0];

        let status = if predicted == expected as i32 { "✓" } else { "✗" };

        println!(
            "  {} + {} mod 97 = {} | Model predicted: {} {}",
            a, b, expected, predicted, status
        );
    }

    println!();
    println!("{}", "=".repeat(60));
    println!("✅ Inference complete!");
}
