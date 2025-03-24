import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name, ylabel):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    # ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    # # Training time
    # single_mean, single_std = 14.678404116630555, 0.13861007498846228
    # device0_mean, device0_std =  7.878487634658813, 0.22188336946174833
    # device1_mean, device1_std =  7.564719223976136, 0.2777229285615978

    # plot([device0_mean, device1_mean, single_mean],
    #     [device0_std, device1_std, single_std],
    #     ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
    #     'training_time.png',
    #     'GPT2 Execution Time (Second)'
    # )

    # # Throughput
    # single_tok_mean, single_tok_std = 258327.0615228204, 1395.9194291237677
    # rank0_tok_mean, rank0_tok_std = 254734.38252730388, 2262.1687238424097
    # rank1_tok_mean, rank1_tok_std = 254718.85885143658, 2125.880021278328

    # ddp_tok_mean = rank0_tok_mean + rank1_tok_mean
    # ddp_tok_std = (rank0_tok_std ** 2 + rank1_tok_std ** 2) ** 0.5

    # plot(
    #     [ddp_tok_mean, single_tok_mean],
    #     [ddp_tok_std, single_tok_std],
    #     ['Data Parallel - 2 GPUs', 'Single GPU'],
    #     'throughput.png',
    #     'GPT2 Throughput (Tokens per second)'
    # )

    # Training time
    # pp_mean, pp_std = 15.628819346427917, 0.11727392673492432
    # mp_mean, mp_std = 17.696326732635498, 2.2995378971099854

    # Throughput
    pp_mean, pp_std = 40952.296097561906, 307.293626326933
    mp_mean, mp_std = 36786.863487661976, 4780.245526868841

    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp.png',
        'GPT2 Execution Time (Second)')