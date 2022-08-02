import torchvision
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from joblib import Parallel
from sklearn.utils.fixes import delayed, _joblib_parallel_args
import gzip
import cv2
import jax.numpy as jnp

import jax
from jax import nn

def compress(filepath, arr, img=False):
    if img: arr = (arr*255).astype(np.uint8)
    f = gzip.GzipFile(filepath, "w")
    np.save(f, arr)
    f.close()

def load(filepath, img=False):
    f = gzip.GzipFile(filepath, "r")
    arr = np.load(f)
    return arr


def get_dataset(config):
    dataset = config.tag
    transform = config.transform_loader
    if dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=config.datadir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=config.datadir, train=False, download=True, transform=transform)
    elif dataset == 'synthetic':
        if not os.path.isdir(config.datadir): os.mkdir(config.datadir)
        trainset = SyntheticDataset(config.image_size, 6, 60000, config.datadir, '/train', transform=transform)
        testset = SyntheticDataset(config.image_size, 6, 10000, config.datadir, '/test', transform=transform)
    return trainset, testset


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inverse_sigmoid_jax(x):
    return -jnp.log(jnp.maximum((1 / x) - 1, 1E-10))

def inverse_sin_cos_jax(sin,cos):
    c = jnp.arccos(cos)
    s = jnp.arcsin(sin)
    return jnp.where(s>0, c, 2*jnp.pi-c)

class SyntheticDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, length, latent, num, fname, folder, transform=None, 
                n_jobs=8, seed=1021, dtype=np.float32, yrange=4, generate=True):
        """
        Args:
            latent (int): '010000' default 6
            num (int)
        """
        self.latent = latent
        self.length = length
        self.num = num
        self.transform = transform
        self.dtype = dtype
        self.yrange = yrange
        self.rng = np.random.default_rng(seed)
        
        if not os.path.isfile(fname+'/background.npy'):
            self.backgrounds = self.rng.uniform(size=(self.latent, length, length))
            for j in range(self.latent):
                self.backgrounds[j] = (self.backgrounds[j] + self.backgrounds[j].T)/2
            np.save(fname+'/background.npy', self.backgrounds)
        else:
            self.backgrounds = np.load(fname+'/background.npy')
        
        s = length // 2
        diag = []
        for i in range(latent):
            v = np.square(self.backgrounds[i][0:s,s:s*2]).sum()/4.
            diag.append(1./v)
            s //= 2
        self.metric = np.diag(diag/np.sqrt(np.square(diag).sum()))
        print(self.metric)
        
        if generate:
            if not os.path.isfile(fname+'/'+folder+'_0_x.npy.gz'):
                print('start generating')
                def generate_batch(size,k):
                    latents, pics, outputs = [],[],[]
                    for i in range(size):
                        latent, pic, output = self.generate()
                        pics.append(pic)
                        outputs.append(output)
                        latents.append(latent)
                    compress(fname+'/'+folder+'_'+str(k)+'_x.npy.gz',np.array(pics),img=True)
                    compress(fname+'/'+folder+'_'+str(k)+'_y.npy.gz',np.array(outputs))
                    compress(fname+'/'+folder+'_'+str(k)+'_z.npy.gz',np.array(latents))
                    return

                data_all = Parallel(n_jobs=n_jobs, verbose=0,
                    **_joblib_parallel_args(prefer='processes'))(
                        delayed(generate_batch)(5000,k) for k in range(self.num//5000)
                    )
            print('loading data')
            pics, outputs, latents= [],[],[]
            for k in range(self.num//5000):
                pics.append(load(fname+'/'+folder+'_'+str(k)+'_x.npy.gz', img=True))
                outputs.append(load(fname+'/'+folder+'_'+str(k)+'_y.npy.gz'))
                latents.append(load(fname+'/'+folder+'_'+str(k)+'_z.npy.gz'))

            self.pics, self.outputs, self.latent_zs = np.concatenate(pics), np.concatenate(outputs), np.concatenate(latents)
            if folder=='/train':
                np.save(fname+'/encode_z.npy', self.latent_zs)
            print('loaded with shape', self.pics.shape, self.outputs.shape, self.latent_zs.shape)
        
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        pic,output = self.pics[idx], self.outputs[idx]

        return torch.from_numpy(pic.astype(np.float32)), output

    def generate(self):
        z = self.rng.uniform(0,2*np.pi,size=self.latent)
        y = self.label(z)
        sin_code = (1. +np.sin(z))/2.
        cos_code = (1. +np.cos(z))/2.

        D, d = self.length, self.latent
        pic = np.zeros((D, D), dtype=self.dtype)

        s = D // 2
        for i in range(d):
            pic[0:s,s:s*2] = sin_code[i] * self.backgrounds[i][0:s,s:s*2]
            pic[s:s*2,0:s] = cos_code[i] * self.backgrounds[i][s:s*2,0:s]
            s //= 2

        x = pic[:,:, np.newaxis]

        return z, x, y

    def label(self, z):
        lower, upper = -self.yrange, self.yrange
        for x in z:
            mid = (lower + upper) / 2
            if x <= np.pi:
                lower = mid
            else:
                upper = mid

        return self.rng.normal(
            loc=(lower + upper) / 2,
            scale=np.abs(upper - lower) / 6
        )

    def _decode_jax(self, z):
        sin_code = (1. +jnp.sin(z))/2.
        cos_code = (1. +jnp.cos(z))/2.

        D, d = self.length, self.latent
        pic = jnp.zeros((D, D), dtype=self.dtype)

        s = D // 2
        for i in range(d):
            pic = pic.at[0:s,s:s*2].set(sin_code[i] * self.backgrounds[i][0:s,s:s*2])
            pic = pic.at[s:s*2,0:s].set(cos_code[i] * self.backgrounds[i][s:s*2,0:s])
            s //= 2
        return pic[:,:, jnp.newaxis]

    def decode_jax(self, z):
        prefix = z.shape[:-1]
        z = jnp.reshape(z, (-1, self.latent))
        x = jax.vmap(self._decode_jax)(z)
        x = jnp.reshape(x, prefix + x.shape[1:])
        return x

    def encode_jax(self, x):
        x = jnp.squeeze(x, axis=-1)
        sin_code, cos_code = [],[]

        s = self.length // 2
        for i in range(self.latent):
            sin_v = jnp.divide(
                jnp.sum(x[..., 0:s, s:s*2],axis=(-1,-2)),
                jnp.sum(self.backgrounds[i][0:s, s:s*2])
            )*2. - 1.
            cos_v = jnp.divide(
                jnp.sum(x[..., s:s*2, 0:s],axis=(-1,-2)),
                jnp.sum(self.backgrounds[i][s:s*2, 0:s])
            )*2. - 1.
            sin_code.append(sin_v)
            cos_code.append(cos_v)
            s //= 2

        sin_code = jnp.stack(sin_code, axis=-1)
        cos_code = jnp.stack(cos_code, axis=-1)
        z = inverse_sin_cos_jax(sin_code, cos_code)
        return z


class SyntheticYFDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, length, latent, num, fname, folder, transform=None, 
                n_jobs=8, seed=1021, dtype=np.float32, yrange=4, generate=True):
        """
        Args:
            latent (int): '010000' default 6
            num (int)
        """
        self.latent = latent
        self.length = length
        self.num = num
        self.transform = transform
        self.dtype = dtype
        self.yrange = yrange
        self.rng = np.random.default_rng(seed)
        
        if not os.path.isfile(fname+'/background.npy'):
            self.backgrounds = self.rng.uniform(size=(self.latent, length, length))
            for j in range(self.latent):
                self.backgrounds[j] = (self.backgrounds[j] + self.backgrounds[j].T)/2
            np.save(fname+'/background.npy', self.backgrounds)
        else:
            self.backgrounds = np.load(fname+'/background.npy')

        grid = self.length * np.linspace(0, 1, 2 * self.latent)
        grid = grid.astype(int)

        diag = []
        for j in range(latent):
            i = self.latent-1-j
            x0, y0 = grid[i], grid[i]
            x1, y1 = grid[-(i+1)], grid[-(i+1)]

            tmp = self.backgrounds[i].copy()
            if i < self.latent-1:
                xx0,yy0 = grid[i+1], grid[i+1]
                xx1,yy1 = grid[-(i+2)], grid[-(i+2)]
                tmp[xx0:xx1,yy0:yy1] = 0.
            v = np.square(tmp[x0:x1,y0:y1]).sum()/2.
            diag.insert(0, 1./v)
        self.metric = np.diag(diag/np.sqrt(np.square(diag).sum()))
        print(self.metric)
        
        
        if generate:
            if not os.path.isfile(fname+'/'+folder+'_0_x.npy.gz'):
                print('start generating')
                def generate_batch(size,k):
                    latents, pics, outputs = [],[],[]
                    for i in range(size):
                        latent, pic, output = self.generate()
                        pics.append(pic)
                        outputs.append(output)
                        latents.append(latent)
                    compress(fname+'/'+folder+'_'+str(k)+'_x.npy.gz',np.array(pics),img=True)
                    compress(fname+'/'+folder+'_'+str(k)+'_y.npy.gz',np.array(outputs))
                    compress(fname+'/'+folder+'_'+str(k)+'_z.npy.gz',np.array(latents))
                    return

                data_all = Parallel(n_jobs=n_jobs, verbose=0,
                    **_joblib_parallel_args(prefer='processes'))(
                        delayed(generate_batch)(5000,k) for k in range(self.num//5000)
                    )
            print('loading data')
            pics, outputs, latents= [],[],[]
            for k in range(self.num//5000):
                pics.append(load(fname+'/'+folder+'_'+str(k)+'_x.npy.gz', img=True))
                outputs.append(load(fname+'/'+folder+'_'+str(k)+'_y.npy.gz'))
                latents.append(load(fname+'/'+folder+'_'+str(k)+'_z.npy.gz'))

            self.pics, self.outputs, self.latent_zs = np.concatenate(pics), np.concatenate(outputs), np.concatenate(latents)
            if folder=='/train':
                np.save(fname+'/encode_z.npy', self.latent_zs)
            print('loaded with shape', self.pics.shape, self.outputs.shape, self.latent_zs.shape)
        
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        pic,output = self.pics[idx], self.outputs[idx]

        return torch.from_numpy(pic.astype(np.float32)), output

    def generate(self):
        z = self.rng.uniform(0,2*np.pi,size=self.latent)
        y = self.label(z)
        sin_code = (1.+np.sin(z))/2.
        cos_code = (1.+np.cos(z))/2.

        D, d = self.length, self.latent
        pic = np.zeros((D, D), dtype=self.dtype)
        grid = D * np.linspace(0, 1, 2 * d)
        grid = grid.astype(int)


        for i in range(d):
            x0, y0 = grid[i], grid[i]
            x1, y1 = grid[-(i+1)], grid[-(i+1)]
            tmp = self.backgrounds[i][x0:x1, y0:y1]
            rows = np.tile(np.arange(x0,x1).reshape(-1,1),(1,x1-x0))
            columns = np.tile(np.arange(y0,y1),(y1-y0,1))
            tmp = np.where(rows>columns, sin_code[i]* tmp, cos_code[i] *tmp)

            pic[x0:x1, y0:y1] = tmp
        x = pic[:,:, np.newaxis]

        return z, x, y

    def label(self, z):
        lower, upper = -self.yrange, self.yrange
        for x in z:
            mid = (lower + upper) / 2
            if x <= np.pi:
                lower = mid
            else:
                upper = mid

        return self.rng.normal(
            loc=(lower + upper) / 2,
            scale=np.abs(upper - lower) / 6
        )

    def _decode_jax(self, z):
        sin_code = (1. + jnp.sin(z))/2.
        cos_code = (1. + jnp.cos(z))/2.

        D, d = self.length, self.latent
        pic = jnp.zeros((D, D), dtype=self.dtype)
        grid = D * np.linspace(0, 1, 2 * d)
        grid = grid.astype(int)



        for i in range(d):
            x0, y0 = grid[i], grid[i]
            x1, y1 = grid[-(i+1)], grid[-(i+1)]
            tmp = self.backgrounds[i][x0:x1, y0:y1]
            rows = jnp.tile(jnp.arange(x0,x1).reshape(-1,1),(1,x1-x0))
            columns = jnp.tile(jnp.arange(y0,y1),(y1-y0,1))
            tmp = jnp.where(rows>columns, sin_code[i]* tmp, cos_code[i] *tmp)

            pic = pic.at[x0:x1, y0:y1].set(tmp)

        return pic[:,:, jnp.newaxis]

    def decode_jax(self, z):
        prefix = z.shape[:-1]
        z = jnp.reshape(z, (-1, self.latent))
        x = jax.vmap(self._decode_jax)(z)
        x = jnp.reshape(x, prefix + x.shape[1:])
        return x

    def encode_jax(self, x):
        x = jnp.squeeze(x, axis=-1)
        sin_code, cos_code = [],[]
        grid = self.length * np.linspace(0, 1, 2 * self.latent)
        grid = grid.astype(int)

        for i in range(self.latent):
            x0, y0 = grid[i], grid[i]
            x1, y1 = grid[i+1], grid[i+1]
            w0, w1 = grid[-(i+1)], grid[-(i+2)]
            z0, z1 = grid[-(i+1)], grid[-(i+2)]

            if i==self.latent-1:
                w1, y1 = self.length//2, self.length//2
                z1, x1 = self.length//2, self.length//2
            sin_v = jnp.divide(
                jnp.sum(x[..., z1:z0, y0:y1],axis=(-1,-2)),
                jnp.sum(self.backgrounds[i][z1:z0, y0:y1])
            ) * 2 - 1.0
            cos_v = jnp.divide(
                jnp.sum(x[..., x0:x1, w1:w0],axis=(-1,-2)),
                jnp.sum(self.backgrounds[i][x0:x1, w1:w0])
            ) * 2 - 1.0
            sin_code.append(sin_v)
            cos_code.append(cos_v)

        sin_code = jnp.stack(sin_code, axis=-1)
        cos_code = jnp.stack(cos_code, axis=-1)
        z = inverse_sin_cos_jax(sin_code, cos_code)
        return z
                

class FaceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.folder))

    def __getitem__(self, idx):
        fname = os.listdir(self.folder)[idx]
        im = cv2.imread(self.folder + fname)
        label = int(fname.split('_')[0])
        return torch.from_numpy(im.astype(np.float32)), label

class ImageData(Dataset):
    def __init__(self, images, labels, indices, transform=None):
        self.images = images
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self,idx):
        img = self.images[self.indices[idx]]
        label = self.labels[self.indices[idx]]
        
        return img, label
    

