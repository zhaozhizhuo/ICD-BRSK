from laatmodel.models.attentions.attention_layer import *


def init_attention_layer(model):

    model.attention = AttentionLayer(args=model.args, size=model.output_size,
                                         level_projection_size=128,
                                         n_labels=model.vocab.all_n_labels(), n_level=model.vocab.n_level())
    linears = []
    projection_linears = []
    for level in range(model.vocab.n_level()):
        level_projection_size = 0 if level == 0 else model.level_projection_size
        linears.append(nn.Linear(model.output_size*2 + level_projection_size,
                                 model.vocab.n_labels(level)))
        projection_linears.append(nn.Linear(model.vocab.n_labels(level), model.level_projection_size, bias=False))
    model.linears = nn.ModuleList(linears)
    model.projection_linears = nn.ModuleList(projection_linears)



def perform_attention(model, all_output, last_output):
    attention_weights = None
    attention_mode = 'label'
    previous_level_projection = None
    if attention_mode is not None:
        weighted_outputs = []
        attention_weights = []
        for level in range(model.vocab.n_level()):
            weighted_output, attention_weight = model.attention(all_output,
                                                                previous_level_projection, label_level=level)
            if attention_mode not in ["label", "caml"]:
                if model.use_dropout:
                    weighted_output = model.dropout(weighted_output)
                weighted_output = model.linears[level](weighted_output)

            previous_level_projection = model.projection_linears[level](
                torch.sigmoid(weighted_output) if attention_mode in ["label", "caml"]
                else torch.softmax(weighted_output, 1))
            previous_level_projection = torch.sigmoid(previous_level_projection)
            weighted_outputs.append(weighted_output)
            attention_weights.append(attention_weight)
    else:
        weighted_outputs = []
        attention_weights = None
        previous_level_projection = None
        for level in range(model.vocab.n_level()):
            if previous_level_projection is not None:
                last_output = [last_output, previous_level_projection]
                last_output = torch.cat(last_output, dim=1)

            output = last_output
            if model.use_dropout:
                output = model.dropout(last_output)

            output = model.linears[level](output)
            weighted_outputs.append(output)
            previous_level_projection = model.projection_linears[level](torch.softmax(output, 1))
    return weighted_outputs, attention_weights
