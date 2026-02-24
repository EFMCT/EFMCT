import astra
import astra.experimental
import torch
import math

class OperatorFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, volume, projector, projection_shape, volume_shape):
            if volume.ndim == 4:
                batch_size = volume.shape[0]
                projection = torch.zeros((batch_size, *projection_shape), dtype=torch.float32, device='cuda')
                for i in range(batch_size):  # Process each batch separately
                    astra.experimental.direct_FP3D(projector, vol=volume[i].contiguous().detach(), proj=projection[i])

            elif volume.ndim ==3:
                projection = torch.zeros(projection_shape, dtype=torch.float32, device='cuda')
                astra.experimental.direct_FP3D(projector, vol=volume.detach(), proj=projection)

            else:
                raise NotImplementedError
            
            ctx.save_for_backward(volume)
            ctx.projector = projector  # Save projector for backward computation
            ctx.volume_shape = volume_shape
            return projection
        
        @staticmethod
        def backward(ctx, grad_output):
            volume, = ctx.saved_tensors
            projector = ctx.projector
            volume_shape = ctx.volume_shape

            if volume.ndim==4:
                batch_size = volume.shape[0]
                grad_volume = torch.zeros((batch_size, *volume_shape), dtype=torch.float32, device='cuda')
                for i in range(batch_size):
                    astra.experimental.direct_BP3D(projector, vol=grad_volume[i], proj=grad_output[i].contiguous().detach())

            elif volume.ndim==3:
                grad_volume = torch.zeros(volume_shape, dtype=torch.float32, device='cuda')
                astra.experimental.direct_BP3D(projector, vol=grad_volume, proj=grad_output.detach())
            
            else:
                raise NotImplementedError
        
            return grad_volume, None, None, None # The second 'None' corresponds to the non-trainable projector

class Operator:
    """A linear tomographic projection operator

    An operator describes and computes the projection from a volume onto a
    projection geometry.
    """

    def __init__(
        self,
        volume_geometry,
        projection_geometry):

        super(Operator, self).__init__()
        self.volume_geometry = volume_geometry
        self.projection_geometry = projection_geometry
        self.projector = astra.create_projector('cuda3d', projection_geometry, volume_geometry)
        self.projection_shape = astra.geom_size(projection_geometry)
        self.volume_shape = astra.geom_size(volume_geometry)
        self.num_angles = len(projection_geometry['ProjectionAngles']) if 'ProjectionAngles' in projection_geometry else len(projection_geometry['Vectors'])
        # Compute C
        y_tmp = torch.ones(self.projection_shape, device='cuda')
        C = self.T(y_tmp)
        C[C < 1e-8] = math.inf
        C.reciprocal_()
        # Compute R
        x_tmp = torch.ones(self.volume_shape, device='cuda')
        R = self(x_tmp)
        R[R < 1e-8] = math.inf
        R.reciprocal_()
        self.R = R
        self.C = C

    def __call__(self, volume):
        return OperatorFunction.apply(volume, self.projector, self.projection_shape, self.volume_shape)
    
    def T(self, projection):
        if projection.ndim==4:
            batch_size = projection.shape[0]
            volume = torch.zeros((batch_size, *self.volume_shape), dtype=torch.float32, device='cuda')
            for i in range(batch_size):
                astra.experimental.direct_BP3D(self.projector, vol=volume[i], proj=projection[i].contiguous().detach())
        
        elif projection.ndim==3:
            volume = torch.zeros(self.volume_shape, dtype=torch.float32, device='cuda')
            astra.experimental.direct_BP3D(self.projector, vol=volume, proj=projection.detach())
        
        else:
            raise NotImplementedError
        
        return volume
    
    def forward(self, volume):
        return self.__call__(volume)
    
    def transpose(self, projection):
        return self.T(projection)
    
    def project(self, volume, projection):
        # calculate (I - C * A^T * R* A)x + C A^T y
        return volume - self.C*self.transpose(self.R*self.forward(volume)) + self.C*self.transpose(self.R*projection)


class NoNoise(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Noiseless forward operator"""
        return data