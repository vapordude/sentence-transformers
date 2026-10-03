//! Classification and Regression heads for Cross-Encoders.

/// Regression head that projects embeddings to a single score.
#[derive(Debug, Clone)]
pub struct RegressionHead {
    pub weights: Vec<f32>, // Flattened weights: [1, in_features]
    pub bias: Option<f32>,
    pub in_features: usize,
}

impl RegressionHead {
    pub fn new(in_features: usize, weights: Vec<f32>, bias: Option<f32>) -> Self {
        assert_eq!(weights.len(), in_features);
        Self {
            weights,
            bias,
            in_features,
        }
    }

    /// Forward pass for a batch of vectors.
    /// `inputs` is flattened: [batch_size, in_features]
    /// Returns a flat vector of scores of length `batch_size`.
    pub fn forward_batched(&self, inputs: &[f32]) -> Vec<f32> {
        let _span = crate::info_span!(
            "regression_head_forward_batched",
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
        let mut output = vec![0.0; batch_size];

        for b in 0..batch_size {
            let mut sum = self.bias.unwrap_or(0.0);
            #[allow(clippy::needless_range_loop)]
            for j in 0..self.in_features {
                sum += inputs[b * self.in_features + j] * self.weights[j];
            }
            output[b] = sum;
        }
        output
    }
}

/// Classification head that projects embeddings to a logits vector per item in a batch.
#[derive(Debug, Clone)]
pub struct ClassificationHead {
    pub weights: Vec<f32>, // Flattened weights: [num_classes, in_features]
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub num_classes: usize,
}

impl ClassificationHead {
    pub fn new(
        in_features: usize,
        num_classes: usize,
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Self {
        assert_eq!(weights.len(), in_features * num_classes);
        if let Some(ref b) = bias {
            assert_eq!(b.len(), num_classes);
        }
        Self {
            weights,
            bias,
            in_features,
            num_classes,
        }
    }

    /// Forward pass for a batch of vectors.
    /// `inputs` is flattened: [batch_size, in_features]
    /// Returns a flat vector of logits of length `batch_size * num_classes`.
    pub fn forward_batched(&self, inputs: &[f32]) -> Vec<f32> {
        let _span = crate::info_span!(
            "classification_head_forward_batched",
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
        let mut output = vec![0.0; batch_size * self.num_classes];

        for b in 0..batch_size {
            for i in 0..self.num_classes {
                let mut sum = if let Some(ref bias) = self.bias {
                    bias[i]
                } else {
                    0.0
                };
                for j in 0..self.in_features {
                    sum +=
                        inputs[b * self.in_features + j] * self.weights[i * self.in_features + j];
                }
                output[b * self.num_classes + i] = sum;
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regression_head() {
        let weights = vec![0.5, -0.5];
        let bias = Some(1.0);
        let head = RegressionHead::new(2, weights, bias);

        let inputs = vec![2.0, 4.0, 1.0, 1.0];
        let output = head.forward_batched(&inputs);

        // b=0: 2*0.5 + 4*(-0.5) + 1.0 = 1.0 - 2.0 + 1.0 = 0.0
        // b=1: 1*0.5 + 1*(-0.5) + 1.0 = 0.5 - 0.5 + 1.0 = 1.0
        assert_eq!(output, vec![0.0, 1.0]);
        println!("ST-EVIDENCE id=st-cross-encoder-heads checked=1 exact=true paths=heads");
    }

    #[test]
    fn test_classification_head() {
        let weights = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2]
        let bias = Some(vec![0.5, -0.5]);
        let head = ClassificationHead::new(2, 2, weights, bias);

        let inputs = vec![2.0, 4.0, 1.0, -1.0];
        let output = head.forward_batched(&inputs);

        // b=0: class 0 = 2*1 + 4*0 + 0.5 = 2.5
        //      class 1 = 2*0 + 4*1 - 0.5 = 3.5
        // b=1: class 0 = 1*1 + (-1)*0 + 0.5 = 1.5
        //      class 1 = 1*0 + (-1)*1 - 0.5 = -1.5
        assert_eq!(output, vec![2.5, 3.5, 1.5, -1.5]);
        println!("ST-EVIDENCE id=st-cross-encoder-heads checked=2 exact=true paths=heads");
    }
}
