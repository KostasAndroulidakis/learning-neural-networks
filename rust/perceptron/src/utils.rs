use crate::perceptron::Perceptron;

pub struct Metrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: ConfusionMatrix,
}

pub struct ConfusionMatrix {
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_: usize,
}

pub fn evaluate_model(y_true: &[usize], y_pred: &[usize]) -> Metrics {
    let mut cm = ConfusionMatrix {
        tp: 0,
        fp: 0,
        tn: 0,
        fn_: 0,
    };
    
    for (i, &true_val) in y_true.iter().enumerate() {
        let pred_val = y_pred[i];
        
        match (true_val, pred_val) {
            (1, 1) => cm.tp += 1,
            (0, 1) => cm.fp += 1,
            (0, 0) => cm.tn += 1,
            (1, 0) => cm.fn_ += 1,
            _ => panic!("Invalid class labels, expected 0 or 1"),
        }
    }
    
    let accuracy = (cm.tp + cm.tn) as f64 / y_true.len() as f64;
    
    let precision = if cm.tp + cm.fp > 0 {
        cm.tp as f64 / (cm.tp + cm.fp) as f64
    } else {
        0.0
    };
    
    let recall = if cm.tp + cm.fn_ > 0 {
        cm.tp as f64 / (cm.tp + cm.fn_) as f64
    } else {
        0.0
    };
    
    let f1_score = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    
    Metrics {
        accuracy,
        precision,
        recall,
        f1_score,
        confusion_matrix: cm,
    }
}

pub fn print_metrics(metrics: &Metrics) {
    println!("Model Performance Metrics:");
    println!("-------------------------");
    println!("Accuracy:  {:.4}%", metrics.accuracy * 100.0);
    println!("Precision: {:.4}", metrics.precision);
    println!("Recall:    {:.4}", metrics.recall);
    println!("F1 Score:  {:.4}", metrics.f1_score);
    println!();
    println!("Confusion Matrix:");
    println!("-----------------");
    println!("True Positives:  {}", metrics.confusion_matrix.tp);
    println!("False Positives: {}", metrics.confusion_matrix.fp);
    println!("True Negatives:  {}", metrics.confusion_matrix.tn);
    println!("False Negatives: {}", metrics.confusion_matrix.fn_);
}

pub fn print_learning_curve(errors: &[usize]) {
    println!("Learning Curve (Misclassifications per Epoch):");
    println!("---------------------------------------------");
    
    if errors.is_empty() {
        println!("No training errors recorded");
        return;
    }
    
    let max_error = *errors.iter().max().unwrap();
    let chart_width = 50;
    
    for (epoch, &error) in errors.iter().enumerate() {
        print!("Epoch {:3}: {:3} ", epoch + 1, error);
        
        let bar_length = (error as f64 / max_error as f64 * chart_width as f64) as usize;
        println!("{}", "*".repeat(bar_length));
    }
    println!();
}

pub fn plot_decision_boundary(x: &[Vec<f64>], y: &[usize], perceptron: &Perceptron) {
    if x[0].len() != 2 {
        println!("Decision boundary visualization requires 2D data.");
        return;
    }
    
    // Determine bounds of the plot
    let mut x_min = x[0][0];
    let mut x_max = x[0][0];
    let mut y_min = x[0][1];
    let mut y_max = x[0][1];
    
    for sample in x {
        x_min = x_min.min(sample[0]);
        x_max = x_max.max(sample[0]);
        y_min = y_min.min(sample[1]);
        y_max = y_max.max(sample[1]);
    }
    
    // Add margin
    let margin_x = (x_max - x_min) * 0.1;
    let margin_y = (y_max - y_min) * 0.1;
    x_min -= margin_x;
    x_max += margin_x;
    y_min -= margin_y;
    y_max += margin_y;
    
    // Create grid
    let grid_size = 20;
    let mut grid = vec![vec![' '; grid_size]; grid_size];
    
    // Plot decision boundary
    let weights = perceptron.get_weights();
    let bias = perceptron.get_bias();
    
    if weights[1] != 0.0 {
        for i in 0..grid_size {
            let x_val = x_min + (x_max - x_min) * i as f64 / (grid_size - 1) as f64;
            let y_val = -(weights[0] * x_val + bias) / weights[1];
            
            // Convert to grid coordinates
            let grid_y = ((y_val - y_min) / (y_max - y_min) * (grid_size - 1) as f64) as isize;
            
            if grid_y >= 0 && grid_y < grid_size as isize {
                grid[(grid_size - 1 - grid_y as usize)][i] = '-';
            }
        }
    }
    
    // Plot data points
    for (i, sample) in x.iter().enumerate() {
        let grid_x = ((sample[0] - x_min) / (x_max - x_min) * (grid_size - 1) as f64) as usize;
        let grid_y = ((sample[1] - y_min) / (y_max - y_min) * (grid_size - 1) as f64) as usize;
        
        if grid_x < grid_size && grid_y < grid_size {
            grid[grid_size - 1 - grid_y][grid_x] = if y[i] == 1 { '+' } else { 'o' };
        }
    }
    
    // Print visualization
    println!("Decision Boundary Visualization:");
    println!("--------------------------------");
    println!("+ : Class 1, o : Class 0, - : Decision Boundary");
    println!();
    
    for row in &grid {
        for &cell in row {
            print!("{} ", cell);
        }
        println!();
    }
    
    println!();
    println!("X-axis: {:.4} to {:.4}", x_min, x_max);
    println!("Y-axis: {:.4} to {:.4}", y_min, y_max);
    println!();
}
