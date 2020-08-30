import sys

"""
If program `gnuplot` is available, set global and import `termplotlib`
"""
def has_gnuplot():
    from shutil import which
    return which('gnuplot') is not None

has_gnuplot = has_gnuplot()

if has_gnuplot:
    import termplotlib as tpl

def log_progress(percent_correct_last_50, loss, num_cycles, t_len):
    avg_loss = round((loss / num_cycles), 3)

    fmt = '{:<8}'
    f_c = fmt.format(f'{percent_correct_last_50} %')
    f_l = fmt.format(f'{avg_loss}')
    progress = f'[ {num_cycles} / {t_len} ]  --  % correct of last 50: {f_c} |  Avg. loss: {f_l}'

    sys.stdout.write(progress)
    sys.stdout.flush()
    sys.stdout.write("\b" * len(progress))

def log_epoch_results(layers, epoch_index, rate, num_correct, loss, t_len, x_training_cycles, y_mean_correct, num_classes):
    print_layers = '\n'
    for layer in layers:
        name = layer.l_name
        print_layers += f'     {name}\n'

    avg_loss = round((loss / t_len), 3)
    percent_correct = round(((num_correct / t_len) * 100), 3)

    print('\n\n---\n')
    print(f'Epoch {epoch_index} summary:\n')
    print(f' Config:')
    print(f'   Learning rate: {rate}')
    print(f'   Layers: {print_layers}')
    print(f' Results:')
    print(f'   Number correct: {num_correct}')
    print(f'   Percent correct: {percent_correct}')
    print(f'   Average loss: {avg_loss}')
    print(f'   Number of classes: {num_classes}')
    print('\n---\n')

    if has_gnuplot:
        fig = tpl.figure()
        fig.plot(x_training_cycles, y_mean_correct, width = 60, height=20, ylim=(0, 100), title='% correct per 50 over time')
        fig.show()

