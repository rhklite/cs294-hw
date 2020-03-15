import torch
import print_custom as db
import unittest


from train_pg_f18 import build_mlp

class ModelTest(unittest.TestCase):

    def testModelOutputShape(self):
        input_size, ouput_size, layers, hidden = 5, 5, 20, 10
        model = build_mlp(input_size, ouput_size, hidden, layers)

        batch = 5
        inputs, outputs = [batch, input_size], [batch, ouput_size]
        out = model(torch.randn(inputs))
        self.assertEqual(list(out.shape), outputs)

if __name__ == "__main__":
    unittest.main()
