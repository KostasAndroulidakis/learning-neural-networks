use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;

pub fn generate_linearly_separable_data(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    random_state: u64,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(random_state);
    
    // Generate random weight vector
    let weights: Vec<f64> = (0..n_features)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);
    
    for _ in 0..n_samples {
        // Generate random features
        let features: Vec<f64> = (0..n_features)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        
        // Compute raw output: w·x
        let mut raw_output = 0.0;
        for (j, &w) in weights.iter().enumerate() {
            raw_output += features[j] * w;
        }
        
        // Add noise
        if noise > 0.0 {
            raw_output += rng.gen_range(-noise..noise);
        }
        
        // Create binary target
        let target = if raw_output > 0.0 { 1 } else { 0 };
        
        x.push(features);
        y.push(target);
    }
    
    (x, y)
}

pub fn train_test_split(
    x: &[Vec<f64>],
    y: &[usize],
    test_size: f64,
    random_state: u64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>, Vec<usize>) {
    let n_samples = x.len();
    let n_test = (n_samples as f64 * test_size) as usize;
    
    // Create shuffle indices
    let mut indices: Vec<usize> = (0..n_samples).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(random_state);
    indices.shuffle(&mut rng);
    
    let test_indices = &indices[0..n_test];
    let train_indices = &indices[n_test..];
    
    let mut x_train = Vec::with_capacity(train_indices.len());
    let mut x_test = Vec::with_capacity(test_indices.len());
    let mut y_train = Vec::with_capacity(train_indices.len());
    let mut y_test = Vec::with_capacity(test_indices.len());
    
    for &i in train_indices {
        x_train.push(x[i].clone());
        y_train.push(y[i]);
    }
    
    for &i in test_indices {
        x_test.push(x[i].clone());
        y_test.push(y[i]);
    }
    
    (x_train, x_test, y_train, y_test)
}

pub fn normalize_data(
    x_train: &[Vec<f64>],
    x_test: Option<&[Vec<f64>]>,
) -> (Vec<Vec<f64>>, Option<Vec<Vec<f64>>>) {
    if x_train.is_empty() {
        return (Vec::new(), None);
    }
    
    let n_features = x_train[0].len();
    
    // Find min and max for each feature
    let mut x_min = vec![f64::MAX; n_features];
    let mut x_max = vec![f64::MIN; n_features];
    
    for sample in x_train {
        for (j, &value) in sample.iter().enumerate() {
            x_min[j] = x_min[j].min(value);
            x_max[j] = x_max[j].max(value);
        }
    }
    
    // Calculate range
    let mut x_range = vec![0.0; n_features];
    for j in 0..n_features {
        x_range[j] = x_max[j] - x_min[j];
        if x_range[j] == 0.0 {
            x_range[j] = 1.0; // Avoid division by zero
        }
    }
    
    // Normalize training data
    let x_train_normalized: Vec<Vec<f64>> = x_train
        .iter()
        .map(|sample| {
            sample
                .iter()
                .enumerate()
                .map(|(j, &value)| (value - x_min[j]) / x_range[j])
                .collect()
        })
        .collect();
    
    // Normalize test data if provided
    let x_test_normalized = x_test.map(|x_test| {
        x_test
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(j, &value)| (value - x_min[j]) / x_range[j])
                    .collect()
            })
            .collect()
    });
    
    (x_train_normalized, x_test_normalized)
}
