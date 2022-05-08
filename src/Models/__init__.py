from .ssm import SSM
from .rnn import RNN
from .rssm import RSSM

MODEL_DICT = {
    "ssm": SSM,
    "rnn": RNN,
    "rssm": RSSM
}
