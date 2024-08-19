import numpy as np
from scipy.spatial.transform import Rotation as R

class DataAugmentation:
    def __init__(self, rotate_prob=0.0):
        self.rotate_prob = rotate_prob
                    
    def rotating(self, traj, prob=1):
        """
        traj: [t_all, J, 3]: apply random rotations with probability 1
        """
        rotation_axes = ['z'] # 'x' and 'y' not used because the person could be upside down
        for a in rotation_axes:
            if np.random.rand() < prob:
                degrees = np.random.randint(0, 360)
                r = R.from_euler(a, degrees, degrees=True).as_matrix().astype(np.float32)
                traj = (r @ traj.reshape((-1, 3)).T).T.reshape(traj.shape)
        return traj

    def __call__(self, sequence):
        """
        seq: [seq_len, J, 3]
        """
        sequence = self.rotating(sequence, prob=self.rotate_prob)

        return sequence
