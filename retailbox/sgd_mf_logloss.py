import tensorflow as tf
import numpy as np


#
# Helpers
#

# Embedding Layer
def embed(inputs, size, dim, name=None):
    # Create Matrix and initialize with random values
    std = np.sqrt(2 / dim)
    emb = tf.Variable(tf.random_uniform([size, dim], -std, std), name=name)

    # Use lookup layer to convert user ot item indexes into vectors
    lookup = tf.nn.embedding_lookup(emb, inputs)
    return lookup

# Import objects


#
# Model Graph
#

# Model Parameters
num_users = uid.max() + 1
num_items = iid.max() + 1
num_factors = 128

lambda_user = 0.0000001
lambda_item = 0.0000001

K = 5
lr = 0.005

graph = tf.Graph()
graph.seed = 1

with graph.as_default():
    place_user = tf.placeholder(tf.int32, shape=(None, 1))
    place_item = tf.placeholder(tf.int32, shape=(None, 1))
    place_y = tf.placeholder(tf.float32, shape=(None, 1))
    
    # Usrs
    user_factors = embed(place_user, num_users, num_factors, "user_factors")
    user_bias = embed(place_user, num_users, 1, "user_bias")
    user_bias = tf.reshape(user_bias, [-1, 1])
    
    # Items
    item_factors = embed(place_item, num_items, num_factors, "item_factors")
    item_bias = embed(place_item, num_items, 1, "item_bias")
    item_bias = tf.reshape(item_bias, [-1, 1])

    global_bias = tf.Variable(0.0, name='global_bias')

    pred = tf.reduce_sum(user_factors * item_factors, axis=2)
    pred = tf.sigmoid(global_bias + user_bias + item_bias + pred)

    reg = lambda_user * tf.reduce_sum(user_factors * user_factors) + \
        lambda_item * tf.reduce_sum(item_factors * item_factors)

    loss = tf.losses.log_loss(place_y, pred)
    loss_total = loss + reg

    opt = tf.train.AdamOptimizer(learning_rate=lr)
    step = opt.minimize(loss_total)

    init = tf.global_variables_initializer()

#
# Model Evaluation
#
def get_variable(graph, session, name):
    v = graph.get_operation_by_name(name)
    v = v.values()[0]
    v = v.eval(session=session)
    return v

def calculate_validation_precision(graph, session, uid):
    U = get_variable(graph, session, 'user_factors')
    I = get_variable(graph, session, 'item_factors')
    bi = get_variable(graph, session, 'item_bias').reshape(-1)

    pred_all = U[uid_val].dot(I.T) + bi
    top_val = (-pred_all).argsort(axis=1)[:, :5]

    imp_baseline = baseline.copy()
    imp_baseline[known_mask] = top_val

    return precision(val_indptr, val_items, imp_baseline)

#
# Training
#
def prepare_batches(seq, step):
    n = len(seq)
    res = []
    for i in range(0, n, step):
        res.append(seq[i:i+step])
    return res

session = tf.Session(config=None, graph=graph)
session.run(init)

np.random.seed(0)

for i in range(10):
    train_idx_shuffle = np.arange(uid.shape[0])
    np.random.shuffle(train_idx_shuffle)
    batches = prepare_batches(train_idx_shuffle, 5000)

    progress = tqdm(total=len(batches))
    for idx in batches:
        pos_samples = len(idx)
        neg_samples = pos_samples * K 

        label = np.concatenate([
                    np.ones(pos_samples, dtype='float32'), 
                    np.zeros(neg_samples, dtype='float32')
                ]).reshape(-1, 1)

        neg_users = np.random.randint(low=0, high=num_users, 
                                    size=neg_samples, dtype='int32')
        neg_items = np.random.randint(low=0, high=num_items,
                                    size=neg_samples, dtype='int32')

        batch_uid = np.concatenate([uid[idx], neg_users]).reshape(-1, 1)
        batch_iid = np.concatenate([iid[idx], neg_items]).reshape(-1, 1)

        feed_dict = {
            place_user: batch_uid,
            place_item: batch_iid,
            place_y: label,
        }
        _, l = session.run([step, loss], feed_dict)
        
        progress.update(1)
        progress.set_description('%.3f' % l)
    progress.close()

    val_precision = calculate_validation_precision(graph, session, uid_val)
    print('epoch %02d: precision: %.3f' % (i+1, val_precision))

# def main():
#     # build_model_graph()


#     print("hello")

# if __name__ == '__main__':
#     main()



