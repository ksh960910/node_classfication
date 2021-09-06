import argparse

#Argument
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--hidden', type=int, default=16, help='number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

args = parser.parse_args()