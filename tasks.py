from abc import ABC, abstractmethod

import ipdb  # noqa
import numpy as np
import sklearn.linear_model as linear_model
import torch
import torchvision

from data_utils import get_IB_data
from models import NnConstructor


class LabelToVec(object):
    """Convert labels  in sample to ONE-HOT vector."""
    def __call__(self, target):
        vec = torch.zeros(10)
        vec[target] = 1.0
        return vec


def gen_r2(pred, y, eps=1e-18):
    """Clipped R square. from 0 to 1."""
    # r square
    if type(pred) is np.ndarray:
        pred = torch.from_numpy(pred).to(y.device)
    u = ((y - pred)**2).sum()
    # ipdb.set_trace()
    v = ((y - y.mean(dim=0))**2).sum()
    r2 = 1 - u / (v + eps)

    # return max(torch.tensor(0.0).to(y.device), r2)
    return r2


class TaskBase(ABC):
    def __init__(self) -> None:
        self.nfa_amp = 1e-3  # amplitude of noise in NFA

    @abstractmethod
    def training_loader(self):
        pass

    @abstractmethod
    def loss_func():
        pass

    @abstractmethod
    def metric_func():
        pass


class RandomBindingTask(TaskBase):
    def __init__(self, n_epochs) -> None:
        super().__init__()
        d = 6
        p = 1
        self.dim_in = d
        self.dim_out = p
        self.lr = 1e-3
        self.n_epochs = n_epochs
        self.d_hiddens = "32-16"
        self.output_act_f = NnConstructor.construct_bounded_tanh(scale=0.5)
        self.task_metric_name = "$R^2$"
        self.backbone_net = None

        self.mi_on_batch = False

        n = 4096

        self.X = np.random.rand(n, d) - 0.5
        self.y = np.random.rand(n, p) - 0.5

        # self.X = self.X.astype(float)
        # self.y = self.y.astype(float)

        reg = linear_model.LinearRegression()
        reg.fit(self.X, self.y)
        print(reg.score(self.X, self.y))

        self.training_set = torch.utils.data.TensorDataset(
            torch.from_numpy(self.X).type(torch.FloatTensor),
            torch.from_numpy(self.y).type(torch.FloatTensor),
        )

    @property
    def training_loader(self):
        return torch.utils.data.DataLoader(
            self.training_set,
            batch_size=len(self.training_set),
        )

    @staticmethod
    def loss_func(pred, y):
        if type(pred) is np.ndarray:
            pred = torch.from_numpy(pred).to(y.device)
        return torch.nn.functional.mse_loss(pred, y)

    @staticmethod
    def metric_func(pred, y, eps=1e-18):
        return gen_r2(pred, y, eps=eps)


class IBTask(TaskBase):
    def __init__(self, n_epochs=1000) -> None:
        super().__init__()
        # training related hyper parameters
        self.dim_in = 12
        self.dim_out = 2
        self.lr = 1e-3
        self.n_epochs = n_epochs
        self.d_hiddens = "8-4"
        self.output_act_f = NnConstructor.construct_sigmoid()
        self.task_metric_name = "Correct rate"
        self.backbone_net = None

        self.mi_on_batch = False

        training_data, test_data = get_IB_data(0)

        self.X = training_data.X
        self.y = training_data.y

        self.training_set = torch.utils.data.TensorDataset(
            torch.from_numpy(training_data.X).type(torch.FloatTensor),
            torch.from_numpy(training_data.Y).type(torch.FloatTensor),
        )
        self.test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(test_data.X).type(torch.FloatTensor),
            torch.from_numpy(test_data.Y).type(torch.FloatTensor),
        )

    @property
    def training_loader(self):
        return torch.utils.data.DataLoader(
            self.training_set,
            batch_size=len(self.training_set),
        )

    @property
    def test_loader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=len(self.test_set),
        )

    @staticmethod
    def loss_func(pred, y):
        if type(pred) is np.ndarray:
            pred = torch.from_numpy(pred).to(y.device)
        # labels = torch.argmax(y, dim=1)
        # return torch.nn.functional.cross_entropy(pred, labels)
        # we still treat this as a prediction error problem.
        return torch.nn.functional.mse_loss(pred, y)

    @staticmethod
    def metric_func(pred, y):
        """Prediction accuracy. from 0 to 1."""
        if type(pred) is np.ndarray:
            pred = torch.from_numpy(pred).to(y.device)

        labels = torch.argmax(y, dim=1)
        return (torch.argmax(pred, dim=1) == labels).float().mean()


class IBRegTask(IBTask):
    def __init__(self, n_epochs=1000) -> None:
        super().__init__(n_epochs=n_epochs)
        # training related hyper parameters
        self.dim_in = 12
        self.dim_out = 1
        self.lr = 1e-3
        self.n_epochs = n_epochs
        self.d_hiddens = "8-4"
        self.output_act_f = NnConstructor.construct_sigmoid()
        self.task_metric_name = "Accuracy"
        self.backbone_net = None

        self.mi_on_batch = True

        training_data, test_data = get_IB_data(0)

        self.X = training_data.X
        self.y = training_data.y.reshape(-1, 1)

        self.training_set = torch.utils.data.TensorDataset(
            torch.from_numpy(training_data.X).type(torch.FloatTensor),
            torch.from_numpy(self.y).type(torch.FloatTensor),
        )
        self.test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(test_data.X).type(torch.FloatTensor),
            torch.from_numpy(test_data.y.reshape(-1,
                                                 1)).type(torch.FloatTensor),
        )

    @staticmethod
    def metric_func(pred, y):
        """Prediction accuracy. from 0 to 1."""
        if type(pred) is np.ndarray:
            pred = torch.from_numpy(pred).to(y.device)

        return ((pred >= 0.5).float() == y).float().mean()


class MNISTTask(TaskBase):
    def __init__(
        self,
        n_epochs,
        d_hidden=128,
        training_batch_size=128,
        test_batch_size=1024,
    ):
        super().__init__()
        self.nfa_amp = 1e-4  # amplitude of noise in NFA
        self.backbone_net = NnConstructor.construct_conv_backbone(d_hidden)
        self.dim_in = d_hidden
        self.d_hiddens = "64-32"
        self.dim_out = 10
        self.lr = 1e-3
        self.n_epochs = n_epochs

        self.mi_on_batch = False

        self.output_act_f = NnConstructor.construct_sigmoid()
        # self.output_act_f = NnConstructor.construct_fixpoint()
        self.task_metric_name = r"Accuracy"

        self.training_batch_size = training_batch_size
        self.training_set = torchvision.datasets.MNIST(
            './data/',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
            ]),
            target_transform=torchvision.transforms.Compose([LabelToVec()]))

        self.test_batch_size = test_batch_size
        self.test_set = torchvision.datasets.MNIST(
            './data/',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307, ), (0.3081, )),
            ]),
            target_transform=torchvision.transforms.Compose([LabelToVec()]))

    @property
    def training_loader(self):
        return torch.utils.data.DataLoader(
            self.training_set,
            batch_size=self.training_batch_size,
        )

    @property
    def test_loader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
        )

    @staticmethod
    def loss_func(pred, y):
        if type(pred) is np.ndarray:
            pred = torch.from_numpy(pred).to(y.device)
        return torch.nn.functional.mse_loss(pred, y)

    @staticmethod
    def metric_func(pred, y):
        """Prediction accuracy. from 0 to 1."""
        if type(pred) is np.ndarray:
            pred = torch.from_numpy(pred).to(y.device)

        labels = torch.argmax(y, dim=1)
        return (torch.argmax(pred, dim=1) == labels).float().mean()


class LabelTo01(object):
    """Convert labels  in sample to ONE-HOT vector."""
    def __call__(self, target):
        return torch.tensor([target / 10.])


class MNISTRegTask(TaskBase):
    def __init__(
        self,
        n_epochs,
        d_hidden=128,
        training_batch_size=128,
        test_batch_size=1024,
    ):
        super().__init__()
        self.nfa_amp = 1e-4  # amplitude of noise in NFA
        self.backbone_net = NnConstructor.construct_conv_backbone(d_hidden)
        self.dim_in = d_hidden
        self.d_hiddens = "64-32"
        self.dim_out = 1
        self.lr = 5e-4
        self.n_epochs = n_epochs

        self.mi_on_batch = False
        # regression so no limits
        self.output_act_f = NnConstructor.construct_fixpoint()

        self.task_metric_name = r"Accuracy"

        self.training_batch_size = training_batch_size
        self.training_set = torchvision.datasets.MNIST(
            './data/',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
            ]),
            target_transform=torchvision.transforms.Compose([
                LabelTo01(),
            ]))

        self.test_batch_size = test_batch_size
        self.test_set = torchvision.datasets.MNIST(
            './data/',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307, ), (0.3081, )),
            ]),
            target_transform=torchvision.transforms.Compose([
                LabelTo01(),
            ]))

    @property
    def training_loader(self):
        return torch.utils.data.DataLoader(
            self.training_set,
            batch_size=self.training_batch_size,
        )

    @property
    def test_loader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
        )

    @staticmethod
    def loss_func(pred, y):
        if type(pred) is np.ndarray:
            pred = torch.from_numpy(pred).to(y.device)
        return torch.nn.functional.mse_loss(pred, y.float())

    @staticmethod
    def metric_func(pred, y):
        """Prediction accuracy. from 0 to 1."""
        if type(pred) is np.ndarray:
            pred = torch.from_numpy(pred).to(y.device)
        return (torch.floor(pred * 10) == torch.floor(y * 10)).float().mean()


class ClippedPolynomialTask(TaskBase):
    def __init__(self,
                 n_epochs,
                 n_d=5120,
                 dim_x=12,
                 dim_y=2,
                 v_limit=5,
                 noise_amp=1.2) -> None:
        super().__init__()
        # training related hyper parameters
        self.dim_in = dim_x
        self.dim_out = dim_y
        self.lr = 5e-3
        self.n_epochs = n_epochs

        self.d_hiddens = "12-12-12"
        # self.d_hiddens = "24"
        # self.output_act_f = lambda x: x

        self.output_act_f = NnConstructor.construct_hardtanh(-v_limit, v_limit)

        self.task_metric_name = r"$R^2$"

        self.backbone_net = None

        self.mi_on_batch = True

        n_training = int(0.8 * n_d)
        X = np.random.normal(0, 1, (n_d, dim_x))
        w = np.random.normal(0, 1, (dim_x, dim_y))
        w1 = np.random.normal(0, 1, (dim_x, dim_y))
        y = np.clip(
            (X**2).dot(w1) + X.dot(w) +
            noise_amp * np.random.normal(0, 1, (n_d, dim_y)),
            -v_limit,
            v_limit,
        )

        self.X = X
        self.y = y

        reg = linear_model.LinearRegression()
        reg.fit(self.X, self.y)
        print(reg.score(self.X, self.y))

        self.training_set = torch.utils.data.TensorDataset(
            torch.from_numpy(X[:n_training]).type(torch.FloatTensor),
            torch.from_numpy(y[:n_training]).type(torch.FloatTensor),
        )
        self.test_set = torch.utils.data.TensorDataset(
            torch.from_numpy(X[n_training:]).type(torch.FloatTensor),
            torch.from_numpy(y[n_training:]).type(torch.FloatTensor),
        )

    @property
    def training_loader(self):
        return torch.utils.data.DataLoader(
            self.training_set,
            # batch_size=len(self.training_set),
            batch_size=256,
        )

    @property
    def test_loader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=len(self.test_set),
        )

    @staticmethod
    def loss_func(pred, y):
        if type(pred) is np.ndarray:
            pred = torch.from_numpy(pred).to(y.device)
        return torch.nn.functional.mse_loss(pred, y)

    @staticmethod
    def metric_func(pred, y, eps=1e-18):
        return gen_r2(pred, y, eps=eps)
