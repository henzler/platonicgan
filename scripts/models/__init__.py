from scripts.models import resnet, default

# add different models here

generator_dict = {
    'default': default.Generator
}

discriminator_dict = {
    'default': default.Discriminator,
    'default_3d': default.Discriminator3d
}

encoder_dict = {
    'default': default.Encoder
}
