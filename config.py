import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def add_bool_arg(parser, name, default=False, help_msg=''):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true', help=help_msg)
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name.replace('-','_'):default})

# Dataloader
data_arg = add_argument_group('Dataloader')
data_arg.add_argument('--input-scale-size', type=int, metavar='N', default=256,
                      help='Images will be scaled to have same length width and depth')
add_bool_arg(data_arg, 'weighted-sampling', True, 'Sample weights using 1/class_size')
data_arg.add_argument('--train-ratio', type=float, metavar='r', default=0.8, help='Ratio of labels used for training')
data_arg.add_argument('--train-batch-size', type=int, default=4, help='Size of the training batches')
data_arg.add_argument('--train-workers', type=int, default=4, help='Number of train workers')
data_arg.add_argument('--test-batch-size', type=int, default=4, help='Size of the testing batches')
data_arg.add_argument('--test-workers', type=int, default=4, help='Number of test workers')

# Training
train_arg = add_argument_group('Training')
train_arg.add_argument('--epochs', type=int, default=0)
train_arg.add_argument('--load-model', type=int, default=0)
train_arg.add_argument('--learning-rate', type=float, default=0.003)
train_arg.add_argument('--momentum', type=float, default=0.9)
train_arg.add_argument('--weight-decay', type=float, default=0.0001)
train_arg.add_argument('--step-size', type=int, default=1)
train_arg.add_argument('--lr-gamma', type=int, default=0.8)
add_bool_arg(train_arg, 'train')

#Testing
test_arg = add_argument_group('Testing')
add_bool_arg(train_arg, 'test')
test_arg.add_argument('--test-model-epoch', type=int, default = 0)

#Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--random-seed', type=int, default=520)
misc_arg.add_argument('--image-path', type=str, default='')
misc_arg.add_argument('--labels-path', type=str, default='')
misc_arg.add_argument('--save-path', type=str, default='')
misc_arg.add_argument('--debug', type=int, default=0)
misc_arg.add_argument('--print-period', type=int, default=30)

#Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--leakyrelu-param', type=float, default=0.01)
net_arg.add_argument('--dropout-rate', type=float, default = 0.1)
net_arg.add_argument('--linear-layers', type=int, default = 4)
net_arg.add_argument('--conv-layers', type=int, default = 6)
add_bool_arg(net_arg, 'skip-layers')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
