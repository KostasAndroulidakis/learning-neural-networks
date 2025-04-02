use perceptron::perceptron::Perceptron;

fn main() {
    println!("=== AND Gate Implementation with Perceptron ===\n");
    
    // Create training data for AND gate
    let x = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    
    let y = vec![0, 0, 0, 1];  // AND gate truth table
    
    // Initialize perceptron with 2 inputs
    let mut perceptron = Perceptron::new(2, 0.1, 100, 42);
    
    // Train the perceptron
    println!("Training perceptron on AND gate...");
    perceptron.fit(&x, &y);
    
    // Display learned weights and bias
    println!("Training completed in {} epochs.", perceptron.get_errors().len());
    print!("Learned weights: [");
    for (i, &weight) in perceptron.get_weights().iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{}", weight);
    }
    println!("]");
    println!("Learned bias: {}\n", perceptron.get_bias());
    
    // Test the perceptron
    println!("Testing AND gate:");
    println!("----------------");
    
    let predictions = perceptron.predict(&x);
    
    for (i, sample) in x.iter().enumerate() {
        println!(
            "Input: [{}, {}] -> Output: {} (Expected: {})",
            sample[0], sample[1], predictions[i], y[i]
        );
    }
}
