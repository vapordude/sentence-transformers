
/// A simple dense layer: `y = xW^T + b`
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// The weight matrix in row-major format. Shape: [out_features, in_features]
    /// We use (out_features, in_features) to match standard PyTorch Linear layer shapes.
    pub weights: Vec<f32>,
    /// The bias vector. Shape: [out_features]
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl DenseLayer {
    /// Creates a new dense layer.
    pub fn new(
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        assert_eq!(
            weights.len(),
            in_features * out_features,
            "weights length must be in_features * out_features"
        );
        if let Some(ref b) = bias {
            assert_eq!(b.len(), out_features, "bias length must match out_features");
        }
        Self {
            weights,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass for a single input vector.
    /// `input` shape: [in_features]
    /// Returns output shape: [out_features]
    pub fn forward_single(&self, input: &[f32]) -> Vec<f32> {
        let _span = info_span!("forward_single", elements = input.len(), fallback = 0).entered();
        assert_eq!(
            input.len(),
            self.in_features,
            "Input size must match in_features"
        );

        let mut output = vec![0.0; self.out_features];
        for i in 0..self.out_features {
            let mut sum = if let Some(ref b) = self.bias {
                b[i]
            } else {
                0.0
            };

            // output[i] = sum(input[j] * weights[i * in_features + j])
            let weight_row = &self.weights[i * self.in_features..(i + 1) * self.in_features];
            for j in 0..self.in_features {
                sum += input[j] * weight_row[j];
            }
            output[i] = sum;
        }
        output
    }

    /// Batched forward pass.
    /// `inputs` is flattened batched input. Shape: [batch_size, in_features]
    /// Returns output shape: [batch_size, out_features]
    pub fn forward_batched(&self, inputs: &[f32]) -> Vec<f32> {
        let batch_size = inputs.len() / self.in_features;
        let _span = info_span!("forward_batched", elements = inputs.len(), fallback = 0).entered();
        assert_eq!(
            inputs.len() % self.in_features,
            0,
            "Inputs length must be a multiple of in_features"
        );

        let mut output = vec![0.0; batch_size * self.out_features];
        for b in 0..batch_size {
            let input_row = &inputs[b * self.in_features..(b + 1) * self.in_features];
            let out_row_start = b * self.out_features;

            for i in 0..self.out_features {
                let mut sum = if let Some(ref bias) = self.bias {
                    bias[i]
                } else {
                    0.0
                };

                let weight_row = &self.weights[i * self.in_features..(i + 1) * self.in_features];
                for j in 0..self.in_features {
                    sum += input_row[j] * weight_row[j];
                }
                output[out_row_start + i] = sum;
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_forward_single() {
        println!("ST-EVIDENCE id=st-models-dense checked=1 exact=true paths=dense");

        let weights = vec![
            1.0, 2.0, // out feature 0
            3.0, 4.0, // out feature 1
            5.0, 6.0, // out feature 2
        ];
        let bias = vec![0.1, 0.2, 0.3];
        let layer = DenseLayer::new(weights, Some(bias), 2, 3);

        let input = vec![0.5, 1.5];
        let out = layer.forward_single(&input);

        // Expected:
        // out[0] = 1.0 * 0.5 + 2.0 * 1.5 + 0.1 = 0.5 + 3.0 + 0.1 = 3.6
        // out[1] = 3.0 * 0.5 + 4.0 * 1.5 + 0.2 = 1.5 + 6.0 + 0.2 = 7.7
        // out[2] = 5.0 * 0.5 + 6.0 * 1.5 + 0.3 = 2.5 + 9.0 + 0.3 = 11.8

        assert_eq!(out.len(), 3);
        assert!((out[0] - 3.6).abs() < 1e-5);
        assert!((out[1] - 7.7).abs() < 1e-5);
        assert!((out[2] - 11.8).abs() < 1e-5);
    }

    #[test]
    fn test_dense_forward_batched() {
        println!("ST-EVIDENCE id=st-models-dense checked=2 exact=true paths=dense");

        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let layer = DenseLayer::new(weights, None, 2, 2);

        let inputs = vec![
            1.0, 2.0, // batch 0
            3.0, 4.0, // batch 1
        ];

        let out = layer.forward_batched(&inputs);

        // Expected batch 0: [1*1 + 2*2, 3*1 + 4*2] = [5.0, 11.0]
        // Expected batch 1: [1*3 + 2*4, 3*3 + 4*4] = [11.0, 25.0]

        assert_eq!(out, vec![5.0, 11.0, 11.0, 25.0]);
    }
}
