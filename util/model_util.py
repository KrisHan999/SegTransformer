import torch
import os


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


# load entire model
def load_checkpoint(model, optimizer, scheduler, ckpt_dir, ckpt_fn, device):
    state_dict = torch.load(os.path.join(ckpt_dir, ckpt_fn), map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    scheduler.load_state_dict(state_dict['scheduler_state_dict'])
    epoch = state_dict['epoch'] + 1
    global_step = state_dict['global_step']
    return model, optimizer, scheduler, epoch, global_step


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, ckpt_dir, ckpt_fn):
    state_dict = {
        'state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'global_step': global_step
    }
    torch.save(state_dict, os.path.join(ckpt_dir, ckpt_fn))


def load_checkpoint_pred(model, ckpt_dir, ckpt_fn, device):
    state_dict = torch.load(os.path.join(ckpt_dir, ckpt_fn), map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    return model


def load_checkpoint_model(model, ckpt_dir, ckpt_fn, device):
    state_dict = torch.load(os.path.join(ckpt_dir, ckpt_fn), map_location=device)
    model.load_state_dict(state_dict['state_dict'])
    return model


# only encoder
def load_checkpoint_encoder(encoder, ckpt_dir, ckpt_fn, device):
    state_dict = torch.load(os.path.join(ckpt_dir, ckpt_fn), map_location=device)
    encoder.load_state_dict(state_dict['state_dict'])
    return encoder


def save_checkpoint_encoder(model, ckpt_dir, ckpt_fn):
    state_dict = {
        'state_dict': model.module.encoder.state_dict(),
    }
    torch.save(state_dict, os.path.join(ckpt_dir, ckpt_fn))


# only decoder
def load_checkpoint_decoder(decoder, ckpt_dir, ckpt_fn, device):
    state_dict = torch.load(os.path.join(ckpt_dir, ckpt_fn), map_location=device)
    decoder.load_state_dict(state_dict['state_dict'])
    return decoder


def save_checkpoint_decoder(model, ckpt_dir, ckpt_fn):
    state_dict = {
        'state_dict': model.module.decoder.state_dict(),
    }
    torch.save(state_dict, os.path.join(ckpt_dir, ckpt_fn))


# freeze
def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


# unfreeze
def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True



