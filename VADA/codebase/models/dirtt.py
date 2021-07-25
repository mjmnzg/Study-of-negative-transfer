import importlib
import tensorflow as tf
import tensorbayes as tb
from codebase.models.extra_layers import basic_accuracy, vat_loss
from codebase.args import args
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_xent_two
from tensorbayes.layers import constant
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2 as softmax_xent
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits as sigmoid_xent

nn = importlib.import_module('codebase.models.nns.{}'.format(args.nn))

def dirtt(input_shape_samples=(None, 2)):
    
    # INPUTS 
    T = tb.utils.TensorDict(dict(
        sess = tf.Session(config=tb.growth_config()),
        src_x = tf.placeholder(tf.float32, shape=input_shape_samples, name='in_src_x'), # source train samples
        src_y = tf.placeholder(tf.int64, shape=(None, args.Y), name="out_src_x"), # source train labels
        trg_x = tf.placeholder(tf.float32, shape=input_shape_samples, name='in_trg_x'), # target train samples
        trg_y = tf.placeholder(tf.int64, shape=(None, args.Y), name="out_trg_x"), # target label samples
        test_x = tf.placeholder(tf.float32, shape=input_shape_samples, name='in_test_x'), # target test samples
        test_y =tf.placeholder(tf.int64, shape=(None, args.Y), name="out_test_y"), # target test labels
        prob = tf.placeholder_with_default(1.0, shape=()),
        phase = tf.placeholder_with_default(True, shape=()),
        learning = tf.Variable(args.lr, name='learning_rate', trainable=False),
        lr = args.lr
    ))
    
    # feature extractor for SOURCE
    src_e, _ = nn.classifier(T.src_x, phase=T.phase, enc_phase=1, trim=args.trim, prob=T.prob)
    # feature extractor for TARGET
    trg_e, _ = nn.classifier(T.trg_x, phase=T.phase, enc_phase=1, trim=args.trim, reuse=True, internal_update=True, prob=T.prob)
    # obtains classifier for SOURCE
    src_p, _ = nn.classifier(src_e, phase=T.phase, enc_phase=0, trim=args.trim, prob=T.prob)
    # share classifier layers for TARGET
    trg_p, _ = nn.classifier(trg_e, phase=T.phase, enc_phase=0, trim=args.trim, reuse=True, internal_update=True, prob=T.prob)
    
    # Classifier SOURCE
    loss_src_class = tf.reduce_mean(softmax_xent(labels=T.src_y, logits=src_p))
    
    # CONDITIONAL ENTROPY LOSS
    loss_trg_cent = tf.reduce_mean(softmax_xent_two(labels=trg_p, logits=trg_p))

    # REPLACING GRADIENT REVERSAL
    # Domain confusion LOSS
    if args.dw > 0 and args.dirt == 0 and (args.loss == "vada" or args.loss == "dann"):
        # DISCRIMINATOR
        # construct first time the discriminator using features of source
        real_logit = nn.feature_discriminator(src_e, phase=True)
        # resusing or sharing discriminator with data of Target
        fake_logit = nn.feature_discriminator(trg_e, phase=True, reuse=True)
        
        # LOSS DISCRIMINATOR
        # Ex~Ds [log (D(f(x)))] + Ex~Dt [log (1-D(f(x)))]
        loss_disc = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.ones_like(real_logit), logits=real_logit) +
            sigmoid_xent(labels=tf.zeros_like(fake_logit), logits=fake_logit))
        
        # LOSS DOMAIN
        # Ex~Ds [log (1-D(f(x)))] + Ex~Dt [log (D(f(x)))]
        loss_domain = 0.5 * tf.reduce_mean(
            sigmoid_xent(labels=tf.zeros_like(real_logit), logits=real_logit) +
            sigmoid_xent(labels=tf.ones_like(fake_logit), logits=fake_logit))

    else:
        # if MODEL does not use LOSS DISCRIMINATOR
        loss_disc = constant(0)
        loss_domain = constant(0)
    
    # Virtual adversarial training (turn off src in non-VADA phase)
    loss_src_vat = vat_loss(T.src_x, src_p, nn.classifier, prob=T.prob) if args.sw > 0 and args.dirt == 0 else constant(0)
    loss_trg_vat = vat_loss(T.trg_x, trg_p, nn.classifier, prob=T.prob) if args.tw > 0 else constant(0)
    
    
    # Evaluation (EMA)
    # This is used to obtain weights averages from all epochs
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    var_class = tf.get_collection('trainable_variables', 'class/')
    ema_op = ema.apply(var_class)
    ema_p, ema_layers = nn.classifier(T.test_x, trim=0, phase=False, reuse=True, getter=tb.tfutils.get_getter(ema), prob=T.prob)
    
    # Teacher model (a back-up of EMA model)
    # THIS IS MODEL USED TO PSEUDO LABEL OR DIRT-T
    teacher_p, teacher_layout = nn.classifier(T.test_x, trim=0, phase=False, scope='teacher', prob=T.prob)
    var_main = tf.get_collection('variables', 'class/(?!.*ExponentialMovingAverage:0)')
    var_teacher = tf.get_collection('variables', 'teacher/(?!.*ExponentialMovingAverage:0)')
    teacher_assign_ops = []
    # copy variables trained on EMA 
    for t, m in zip(var_teacher, var_main):
        ave = ema.average(m)
        ave = ave if ave else m
        teacher_assign_ops += [tf.assign(t, ave)]
    # operation to update TEACHER MODEL
    update_teacher = tf.group(*teacher_assign_ops)
    # probabilities of teacher, it is used to label samples of target domain
    teacher = tb.function(T.sess, [T.test_x], tf.nn.softmax(teacher_p))

    
    # Obtains Accuracies
    src_acc = basic_accuracy(T.src_y, src_p)
    trg_acc = basic_accuracy(T.trg_y, trg_p)
    t_p, _ = nn.classifier(T.test_x, trim=0, phase=False, reuse=True, prob=T.prob)
    t_acc = basic_accuracy(T.test_y, t_p)
    #t_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(t_p, -1), tf.argmax(T.test_y, -1)), tf.float32))
    fn_trg_acc = tb.function(T.sess, [T.test_x, T.test_y], t_acc)
    #ema_acc = basic_accuracy(T.test_y, ema_p)
    ema_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ema_p, -1), tf.argmax(T.test_y, -1)), tf.float32))
    fn_ema_acc = tb.function(T.sess, [T.test_x, T.test_y], ema_acc)
    
    # Optimizer
    dw = constant(args.dw) if args.dirt == 0 else constant(0)
    cw = constant(1)       if args.dirt == 0 else constant(args.bw)
    sw = constant(args.sw) if args.dirt == 0 else constant(0)
    tw = constant(args.tw)
    
        
    # VADA LOSS
    loss_main = (dw * loss_domain +
                cw * loss_src_class +
                sw * loss_src_vat +
                tw * loss_trg_cent +
                tw * loss_trg_vat)
    
    var_main = tf.get_collection('trainable_variables', 'class')

    # LOSS MAIN - part I
    if args.optimizer == "adam":
        train_main = tf.train.AdamOptimizer(T.learning).minimize(loss_main, var_list=var_main)
    elif args.optimizer == "sgd":
        train_main = tf.train.MomentumOptimizer(T.learning, 0.9, use_nesterov=True).minimize(loss_main, var_list=var_main)
    elif args.optimizer == 'rmsprop':
        train_main = tf.train.RMSPropOptimizer(T.learning, 0.9).minimize(loss_main, var_list=var_main)
    else:
        raise Exception("Unknown optimizer %s." % args.optimizer)
    
    train_main = tf.group(train_main, ema_op)
    
    
    # LOSS DISC - part II discriminator
    if args.dw > 0 and args.dirt == 0 and (args.loss == "vada" or args.loss == "dann"):
        var_disc = tf.get_collection('trainable_variables', 'disc')
        
        if args.optimizer == "adam":
            train_disc = tf.train.AdamOptimizer(T.learning, 0.9).minimize(loss_disc, var_list=var_disc)
        elif args.optimizer == "sgd":
            train_disc = tf.train.MomentumOptimizer(T.learning, 0.9).minimize(loss_disc, var_list=var_disc)
        elif args.optimizer == 'rmsprop':
            train_disc = tf.train.RMSPropOptimizer(T.learning, 0.9).minimize(loss_disc, var_list=var_disc)
        else:
            raise Exception("Unknown optimizer %s." % args.optimizer)
        
    else:
        train_disc = constant(0)
    
    
    # Summarizations TENSORBOARD
    summary_disc = [tf.summary.scalar('domain/loss_disc', loss_disc),]
    summary_main = [tf.summary.scalar('domain/loss_domain', loss_domain),
                    tf.summary.scalar('class/loss_src_class', loss_src_class),
                    tf.summary.scalar('hyper/dw', dw),
                    tf.summary.scalar('hyper/cw', cw),
                    tf.summary.scalar('hyper/sw', sw),
                    tf.summary.scalar('hyper/tw', tw),
                    tf.summary.scalar('acc/src_acc', src_acc),
                    tf.summary.scalar('acc/trg_acc', trg_acc)]
    
    # Merge summaries TENSORBOARD
    summary_disc = tf.summary.merge(summary_disc)
    summary_main = tf.summary.merge(summary_main)

    # Saved ops
    c = tf.constant
    T.ops_print = [c('disc'), loss_disc,
                   c('domain'), loss_domain,
                   c('class'), loss_src_class]
    T.ops_disc = [summary_disc, train_disc]
    T.ops_main = [summary_main, train_main]
    T.fn_trg_acc = fn_trg_acc
    T.fn_p = tf.argmax(t_p, -1)
    T.fn_ema_acc = fn_ema_acc # it is used to obtain accuracy in testing
    T.fn_ema_p = tf.argmax(ema_p, -1) # it is used to obtain classes in testing
    T.ema_layers = ema_layers # EMA layout of architecture
    T.teacher = teacher # it is used to obtain teacher model
    T.update_teacher = update_teacher
    T.teacher_layout = teacher_layout # teacher layout of architecture
    
    
    return T
