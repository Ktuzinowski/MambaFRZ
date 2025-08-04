import torch

class MLP(torch.nn.Module):
  def __init__(self, num_i, num_h, num_o):
    super(MLP, self).__init__()
    self.linear1 = torch.nn.Linear(num_i, num_h)
    self.relu = torch.nn.ReLU()
    self.linear2 = torch.nn.Linear(num_h, num_h)
    self.relu2 = torch.nn.ReLU()
    self.linear3 = torch.nn.Linear(num_h, num_o)

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.relu2(x)
    x = self.linear3(x)
    return x


class MLP2(torch.nn.Module):
    def __init__(self, num_i, num_h, num_o):
        super(MLP2, self).__init__()
        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        # x = self.softmax(x) # Remove softmax here because CrossEntropyLoss already has it
        return x

class AttentionModule(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kqv_size):
      super(AttentionModule, self).__init__()
      self.MLPk = MLP(input_size, hidden_size, kqv_size)
      self.MLPq = MLP(input_size, hidden_size, kqv_size)
      self.MLPv = MLP(input_size, hidden_size, kqv_size)
      self.softmax = torch.nn.Softmax()
      self.MLPo = MLP2(kqv_size, 32, 2)

    def forward(self, x):
      window_size = 30
      k_list = list()
      v_list = list()
      a_list = list()
      z_list = list()

      # Divide it into chunks
      x = torch.chunk(x, window_size, dim=1)

      for i in range(window_size - 1):
          k_list.append(self.MLPk(x[i]))
          v_list.append(self.MLPv(x[i]))
      q = self.MLPq(x[window_size - 1])

      # Convert it to 1-dimension tensor and concatenate it
      for i in range(window_size - 1):
          a_list.append(torch.sum(torch.mul(q, k_list[i])).view(1))

      a = a_list[0]
      # Concat all the tensor
      for index, a_item in enumerate(a_list):
          if index == 0:
              continue
          a = torch.cat((a, a_item), 0)

      # Do the softmax
      a = self.softmax(a)
      a_output = torch.split(a, 1)
      for i in range(len(a_output)):
          z_list.append(torch.mul(a_output[i], v_list[i]))

      z = z_list[0]
      for index, z_item in enumerate(z_list):
          if index == 0:
              continue
          z = torch.add(z, z_item)

      # Get the digit
      output = self.MLPo(z)
      return output


def initialize_smartfrz_predictor(in_channel, hid_channel, out_channel):
  return AttentionModule(in_channel, hid_channel, out_channel)