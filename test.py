import CrossCoder
import torch 
import CrossCoderDataset

if __name__ == '__main__':
    LATENT_DIM=900
    LAMBDA_SPARSE=2
    BATCH_SIZE_CROSS = 64
    TRAINING_SIZE_CROSS   = 0.7
    VALIDATION_SIZE_CROSS = 0.1 # smaller validation because we just have to tune the latent_dim hyperparam
    TEST_SIZE_CROSS       = 0.2

    ACTIVATIONS_POKEMON_PATH = './activations_layer4/pokemon'
    ACTIVATIONS_DICE_PATH = './activations_layer4/dice'
    ACTIVATIONS_INTERPOLATED_PATH = './activations_layer4/interpolated'

    model_names = ['dice', 'pokemon', 'merged']
    ds = CrossCoderDataset.CrossCoderDataset(ACTIVATIONS_POKEMON_PATH, ACTIVATIONS_DICE_PATH, ACTIVATIONS_INTERPOLATED_PATH)
    n_acts = ds.get_n_activations()

    # Crosscoder dataset with Interpolation merging technique
    ACTIVATIONS_INTERPOLATED_PATH = './activations_layer4/interpolate'
    interp = CrossCoder.CrossCoder(LATENT_DIM, n_acts, LAMBDA_SPARSE)
    interp.load_state_dict(torch.load('./models/crosscoder/interpolated/model_weights.pth', weights_only=True))
    interp.eval()

    min, max = interp.W_enc.min(), interp.W_enc.max()
    interp.W_enc.data = (interp.W_enc - min)/(max - min)
    print(interp.W_enc.min(), interp.W_enc.max())