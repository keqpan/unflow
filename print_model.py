from options.test_options import TestOptions
from models import create_model
import torch
if __name__ == '__main__':
    opt = TestOptions().parse()
    torch.cuda.set_device(opt.gpu_ids[0])
    model = create_model(opt)      # create a model given opt.model and other options
    model.print_networks()
#     model.setup()
    print(model)

