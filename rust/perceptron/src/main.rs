mod activation;
mod data_handler;
mod perceptron;
mod utils;

use perceptron::Perceptron;
use utils::{evaluate_model, print_learning_curve, print_metrics, plot_decision_boundary};

fn main() {
    // Set random seed for reproducibility
    let random_seed = 42;
    
    println!("=== Perceptron Implementation in Rust ===\n");
    
    // Generate synthetic dataset
    println!("Generating linearly separable dataset...");
    let (x, y) = data_handler::generate_linearly_separable_data(100, 2, 0.1, random_seed);
    
    // Split into training and test sets
    println!("Splitting data into training and test sets...");
    let (x_train, x_test, y_train, y_test) = data_handler::train_test_split(&x, &y, 0.3, random_seed);
    
    // Normalize features
    println!("Normalizing features...");
    let (x_train_norm, x_test_norm_option) = data_handler::normalize_data(&x_train, Some(&x_test));
    let x_test_norm = x_test_norm_option.unwrap();
    
    // Initialize and train the perceptron
    println!("Initializing perceptron...");
    let mut perceptron = Perceptron::new(2, 0.01, 100, random_seed);
    
    println!("Training perceptron...");
    perceptron.fit(&x_train_norm, &y_train);
    
    // Display training results
    let errors = perceptron.get_errors();
    println!("Training completed in {} epochs.", errors.len());
    print!("Final weights: [");
    for (i, &weight) in perceptron.get_weights().iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.4}", weight);
    }
    println!("]");
    println!("Final bias: {:.4}\n", perceptron.get_bias());
    
    // Make predictions on test set
    println!("Making predictions on test set...");
    let y_pred = perceptron.predict(&x_test_norm);
    
    // Evaluate model performance
    println!("Evaluating model performance...\n");
    let metrics = evaluate_model(&y_test, &y_pred);
    print_metrics(&metrics);
    
    // Print learning curve
    print_learning_curve(errors);
    
    // Visualize decision boundary
    plot_decision_boundary(&x_train_norm, &y_train, &perceptron);
}
