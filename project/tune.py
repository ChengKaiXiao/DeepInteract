import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn

from project.datasets.PICP.picp_dgl_data_module import PICPDGLDataModule
from project.utils.deepinteract_constants import NODE_COUNT_LIMIT, RESIDUE_COUNT_LIMIT
from project.utils.mymodel import LitGINI
from project.utils.deepinteract_utils import collect_args, process_args, construct_pl_logger

from pytorch_lightning.callbacks import TQDMProgressBar

import torch
torch.multiprocessing.set_sharing_strategy('file_system')


from project.datasets.DB5.db5_dgl_data_module import DB5DGLDataModule
# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------


def main(args):
    # -----------
    # Data
    # -----------
    # Load protein interface contact prediction (PICP) data module
    picp_data_module = PICPDGLDataModule(casp_capri_data_dir=args.casp_capri_data_dir,
                                         db5_data_dir=args.db5_data_dir,
                                         dips_data_dir=args.dips_data_dir,
                                         batch_size=args.batch_size,
                                         num_dataloader_workers=args.num_workers,
                                         knn=args.knn,
                                         self_loops=args.self_loops,
                                         pn_ratio=args.pn_ratio,
                                         casp_capri_percent_to_use=args.casp_capri_percent_to_use,
                                         db5_percent_to_use=args.db5_percent_to_use,
                                         dips_percent_to_use=args.dips_percent_to_use,
                                         training_with_db5=True,
                                         testing_with_casp_capri=False, #args.testing_with_casp_capri,
                                         process_complexes=args.process_complexes,
                                         input_indep=args.input_indep)
    # picp_data_module.setup()

    db5_data_module = DB5DGLDataModule(data_dir=args.db5_data_dir, 
        batch_size=1, 
        num_dataloader_workers=1, 
        knn=args.knn,
        self_loops=args.self_loops, 
        percent_to_use=args.db5_percent_to_use, 
        process_complexes=args.process_complexes, 
        input_indep=args.input_indep,
        )
    db5_data_module.setup()
    # ------------
    # Fine-Tuning
    # ------------
    # ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    # ckpt_path_exists = os.path.exists(ckpt_path)
    # ckpt_provided = args.ckpt_name != '' and ckpt_path_exists

    ckpt_path = Path('/home/ubuntu/project/DeepInteract/project/Model_128_0/version_None/checkpoints/LitGINI-epoch=08-val_ce=0.019.ckpt')

    # ------------
    # Model
    # ------------
    # Assemble a dictionary of model arguments
    dict_args = vars(args)
    use_wandb_logger = args.logger_name.lower() == 'wandb'  # Determine whether the user requested to use WandB

    # Pick model and supply it with a dictionary of arguments
    # Baseline Model - Geometry-Focused Inter-Graph Node Interaction (GINI)
    model = LitGINI(num_node_input_feats=picp_data_module.dips_test.num_node_features,
                    num_edge_input_feats=picp_data_module.dips_test.num_edge_features,
                    gnn_activ_fn=nn.SiLU(),
                    num_classes=picp_data_module.dips_test.num_classes,
                    max_num_graph_nodes=NODE_COUNT_LIMIT,
                    max_num_residues=RESIDUE_COUNT_LIMIT,
                    testing_with_casp_capri=dict_args['testing_with_casp_capri'],
                    training_with_db5=dict_args['training_with_db5'],
                    pos_prob_threshold=0.5,
                    num_gnn_hidden_channels=dict_args['num_gnn_hidden_channels'],
                    num_gnn_attention_heads=dict_args['num_gnn_attention_heads'],
                    knn=dict_args['knn'],
                    interact_module_type=dict_args['interact_module_type'],
                    num_interact_layers=dict_args['num_interact_layers'],
                    num_interact_hidden_channels=dict_args['num_interact_hidden_channels'],
                    use_interact_attention=dict_args['use_interact_attention'],
                    num_interact_attention_heads=dict_args['num_interact_attention_heads'],
                    num_epochs=dict_args['num_epochs'],
                    pn_ratio=dict_args['pn_ratio'],
                    dropout_rate=dict_args['dropout_rate'],
                    metric_to_track=dict_args['metric_to_track'],
                    weight_decay=dict_args['weight_decay'],
                    batch_size=dict_args['batch_size'],
                    lr=1e-5,#dict_args['lr'],
                    pad=dict_args['pad'],
                    use_wandb_logger=use_wandb_logger,
                    weight_classes=dict_args['weight_classes'],
                    fine_tune=True,
                    ckpt_path=ckpt_path)
    args.experiment_name = f'LitGINI-b{args.batch_size}-gl{args.num_gnn_layers}' \
                           f'-n{args.num_gnn_hidden_channels}' \
                           f'-e{args.num_gnn_hidden_channels}' \
                           f'-il{args.num_interact_layers}-i{args.num_interact_hidden_channels}' \
        if not args.experiment_name \
        else args.experiment_name
    litgini_template_ckpt_filename_metric_to_track = f'{args.metric_to_track}:.3f'
    template_ckpt_filename = 'LitGINI-{epoch:02d}-{' + litgini_template_ckpt_filename_metric_to_track + '}'
    
    # ------------
    # Trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # -------------
    # Learning Rate
    # -------------
    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(model, datamodule=picp_data_module)  # Run learning rate finder
        fig = lr_finder.plot(suggest=True)  # Plot learning rates
        fig.savefig('optimal_lr.pdf')
        fig.show()
        model.hparams.lr = lr_finder.suggestion()  # Save optimal learning rate
        logging.info(f'Optimal learning rate found: {model.hparams.lr}')

    # ------------
    # Logger
    # ------------
    pl_logger = construct_pl_logger(args)  # Log everything to an external logger
    trainer.logger = pl_logger  # Assign specified logger (e.g. TensorBoardLogger) to Trainer instance

    # -----------
    # Callbacks
    # -----------
    # Create and use callbacks
    mode = 'min' if 'ce' in args.metric_to_track else 'max'
    early_stop_callback = pl.callbacks.EarlyStopping(monitor=args.metric_to_track,
                                                     mode=mode,
                                                     min_delta=args.min_delta,
                                                     patience=args.patience)
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor=args.metric_to_track,
        mode=mode,
        verbose=True,
        save_last=True,
        save_top_k=3,
        filename=template_ckpt_filename  # Warning: May cause a race condition if calling trainer.test() with many GPUs
    )
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True)

    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks = [early_stop_callback, ckpt_callback, progress_bar]
    if args.fine_tune:
        callbacks.append(lr_monitor_callback)
    trainer.callbacks = callbacks

    # ------------
    # Restore
    # ------------
    # If using WandB, download checkpoint artifact from their servers if the checkpoint is not already stored locally
    
    # model = model.load_from_checkpoint(ckpt_path,
    #                                     use_wandb_logger=use_wandb_logger,
    #                                     batch_size=args.batch_size,
    #                                     lr=1e-5,
    #                                     weight_decay=args.weight_decay)

    # -------------
    # Training
    # -------------
    # Train with the provided model and DataModule
    trainer.fit(model=model, datamodule=db5_data_module)

    # -------------
    # Testing
    # -------------
    trainer.test()


if __name__ == '__main__':
    # -----------
    # Arguments
    # -----------
    # Collect all arguments
    parser = collect_args()

    # Parse all known and unknown arguments
    args, unparsed_argv = parser.parse_known_args()

    # Let the model add what it wants
    parser = LitGINI.add_model_specific_args(parser)

    # Re-parse all known and unknown arguments after adding those that are model specific
    args, unparsed_argv = parser.parse_known_args()

    # Set Lightning-specific parameter values before constructing Trainer instance
    args.max_time = {'hours': args.max_hours, 'minutes': args.max_minutes}
    args.max_epochs = args.num_epochs
    args.profiler = args.profiler_method
    # args.accelerator = args.multi_gpu_backend
    # args.strategy = args.train_strategy
    args.auto_select_gpus = args.auto_choose_gpus
    args.gpus = args.num_gpus
    # args.num_nodes = args.num_compute_nodes
    args.precision = args.gpu_precision
    args.accumulate_grad_batches = args.accum_grad_batches
    args.gradient_clip_val = args.grad_clip_val
    args.gradient_clip_algo = args.grad_clip_algo
    args.stochastic_weight_avg = args.stc_weight_avg
    args.deterministic = True  # Make LightningModule's training reproducible


    # Finalize all arguments as necessary
    args = process_args(args)

    # Begin execution of model training with given args
    main(args)
