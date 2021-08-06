import time
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from options.train_options import TrainOptions


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training samples = %d' % dataset_size)
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0
    total_batch_iters = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            total_batch_iters += 1
            model.set_input(data)
            model.optimize_parameters()
            if total_batch_iters % opt.display_freq == 0:
                visuals = model.get_current_visuals()
                losses = model.get_current_losses()
                visualizer.display_current_results(visuals, epoch, total_iters % opt.update_html_freq == 0)
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            if total_batch_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
