use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::iter::zip;

pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    max_epochs: usize,
    errors: Vec<usize>,
    rng: ChaCha8Rng,
}

impl Perceptron {
    pub fn new(n_inputs: usize, learning_rate: f64, max_epochs: usize, random_state: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(random_state);
        
        // Initialize weights with small random values
        let weights = (0..n_inputs)
            .map(|_| rng.gen_range(-0.01..0.01))
            .collect();
            
        Self {
            weights,
            bias: 0.0,
            learning_rate,
            max_epochs,
            errors: Vec::new(),
            rng,
        }
    }
    
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[usize]) -> &mut Self {
        self.errors.clear();
        
        for _ in 0..self.max_epochs {
            let y_pred = self.predict(x);
            let misclassified = self.update_weights(x, y, &y_pred);
            self.errors.push(misclassified);
            
            if misclassified == 0 {
                break;
            }
        }
        
        self
    }
    
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<usize> {
        x.iter()
            .map(|sample| {
                // Calculate net input: w·x + b
                let net_input = zip(sample.iter(), self.weights.iter())
                    .map(|(xi, wi)| xi * wi)
                    .sum::<f64>() + self.bias;
                    
                // Apply step function
                if net_input >= 0.0 { 1 } else { 0 }
            })
            .collect()
    }
    
    fn update_weights(&mut self, x: &[Vec<f64>], y: &[usize], y_pred: &[usize]) -> usize {
        let mut misclassified = 0;
        
        for (i, sample) in x.iter().enumerate() {
            // Convert prediction to signed error
            let error = y[i] as isize - y_pred[i] as isize;
            
            if error != 0 {
                misclassified += 1;
                
                // Update bias
                self.bias += self.learning_rate * error as f64;
                
                // Update weights
                for (j, xi) in sample.iter().enumerate() {
                    self.weights[j] += self.learning_rate * error as f64 * xi;
                }
            }
        }
        
        misclassified
    }
    
    pub fn get_errors(&self) -> &Vec<usize> {
        &self.errors
    }
    
    pub fn get_weights(&self) -> &Vec<f64> {
        &self.weights
    }
    
    pub fn get_bias(&self) -> f64 {
        self.bias
    }
}
