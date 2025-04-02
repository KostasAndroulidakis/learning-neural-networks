pub fn step(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

pub fn step_derivative(_x: f64) -> f64 {
    0.0 // Technically undefined at 0, but 0 elsewhere
}

pub fn sigmoid(x: f64) -> f64 {
    // Clip to avoid overflow
    let x_safe = x.max(-500.0).min(500.0);
    1.0 / (1.0 + (-x_safe).exp())
}

pub fn sigmoid_derivative(x: f64) -> f64 {
    let sig = sigmoid(x);
    sig * (1.0 - sig)
}

pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn tanh_derivative(x: f64) -> f64 {
    let tanh_x = x.tanh();
    1.0 - tanh_x * tanh_x
}

// Vector versions
pub fn step_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| step(xi)).collect()
}

pub fn sigmoid_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| sigmoid(xi)).collect()
}

pub fn relu_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| relu(xi)).collect()
}

pub fn tanh_vec(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| tanh(xi)).collect()
}
