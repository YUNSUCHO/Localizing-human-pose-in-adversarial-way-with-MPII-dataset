"""
Specifies default parameters used for training in a dictionary
"""

config = {}

hourglass_params = {}
hourglass_params['num_reductions'] = 4
hourglass_params['num_residual_modules'] = 2

dataset = {}
dataset['num_joints'] = 16
#it was 16 in place of 14  #14 for lsp and #16 for mpii #please refer the no of joint cordinates present in the datatsets that you take

pose_discriminator = {}
# TODO: base case now; add 3 later for in_channels
pose_discriminator['in_channels'] = 16
pose_discriminator['num_channels'] = 128
# pose_discriminator['num_joints'] = dataset['num_joints']
pose_discriminator['num_residuals'] = 5 # originally it is 5

generator = {}
# generator['num_joints'] = dataset['num_joints']
generator['num_stacks'] = 4 # orginally it is 4
generator['hourglass_params'] = hourglass_params
generator['mid_channels'] = 512
generator['preprocessed_channels'] = 64

training = {}
training['gen_iters'] = 1
training['disc_iters'] = 1 ## originally it was 1
training['alpha'] = 1.0 / 220
training['beta']  = 1.0 / 180
# training['alpha'] = 0.1

config = {'dataset': dataset, 'generator': generator, 'discriminator': pose_discriminator, 
            'training': training}