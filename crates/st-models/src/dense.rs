/// A simple dense layer: `y = xW^T + b`
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Vec<f32>, // Flattened weights: [out_features, in_features]
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl DenseLayer {
    pub fn new(
        in_features: usize,
        out_features: usize,
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Self {
        assert_eq!(weights.len(), in_features * out_features);
        if let Some(ref b) = bias {
            assert_eq!(b.len(), out_features);
        }
        Self {
            weights,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass for a single vector.
    pub fn forward_single(&self, input: &[f32]) -> Vec<f32> {
        let _span = crate::info_span!("dense_forward_single", elements = input.len(), fallback = 0)
            .entered();
        assert_eq!(input.len(), self.in_features);
        let mut output = vec![0.0; self.out_features];

        for i in 0..self.out_features {
            let mut sum = if let Some(ref b) = self.bias {
                b[i]
            } else {
                0.0
            };
            #[allow(clippy::needless_range_loop)]
            for j in 0..self.in_features {
                sum += input[j] * self.weights[i * self.in_features + j];
            }
            output[i] = sum;
        }
        output
    }

    /// Forward pass for a batch of vectors.
    /// `inputs` is flattened: [batch_size, in_features]
    pub fn forward_batched(&self, inputs: &[f32]) -> Vec<f32> {
        let _span = crate::info_span!(
            "dense_forward_batched",
            elements = inputs.len(),
            fallback = 0
        )
        .entered();
        #[allow(clippy::manual_is_multiple_of)]
        let is_multiple = inputs.len() % self.in_features == 0;
        assert!(
            is_multiple,
            "Input length must be a multiple of in_features"
        );

        let batch_size = inputs.len() / self.in_features;
        let mut output = vec![0.0; batch_size * self.out_features];

        for b in 0..batch_size {
            for i in 0..self.out_features {
                let mut sum = if let Some(ref bias) = self.bias {
                    bias[i]
                } else {
                    0.0
                };
                for j in 0..self.in_features {
                    sum +=
                        inputs[b * self.in_features + j] * self.weights[i * self.in_features + j];
                }
                output[b * self.out_features + i] = sum;
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
        let weights = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let bias = Some(vec![0.5, 0.5]);
        let layer = DenseLayer::new(2, 2, weights, bias);

        let input = vec![1.0, 1.0];
        let output = layer.forward_single(&input);

        // y_0 = 1*1 + 1*2 + 0.5 = 3.5
        // y_1 = 1*3 + 1*4 + 0.5 = 7.5
        assert_eq!(output, vec![3.5, 7.5]);
        println!("ST-EVIDENCE id=st-models-dense checked=1 exact=true paths=dense");
    }

    #[test]
    fn test_dense_forward_batched() {
        let weights = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let bias = Some(vec![0.5, 0.5]);
        let layer = DenseLayer::new(2, 2, weights, bias);

        let inputs = vec![1.0, 1.0, 2.0, 2.0];
        let output = layer.forward_batched(&inputs);

        // b=0: [3.5, 7.5]
        // b=1: y_0 = 2*1 + 2*2 + 0.5 = 6.5
        //      y_1 = 2*3 + 2*4 + 0.5 = 14.5
        assert_eq!(output, vec![3.5, 7.5, 6.5, 14.5]);
        println!("ST-EVIDENCE id=st-models-dense checked=2 exact=true paths=dense");
    }
}
