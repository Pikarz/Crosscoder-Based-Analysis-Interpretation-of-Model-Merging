import torchvision
import re 

if __name__ == '__main__':
    regex_activations = '^layer[1, 3]$'

    test = torchvision.models.resnet50()
    for name, module in test.named_modules():
        if bool( re.search(regex_activations, name) ):
            print(name)
  