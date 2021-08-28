import tensorflow.keras.backend as K


# https://github.com/aldi-dimara/keras-focal-loss/blob/master/focal_loss.py
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        return K.sum(weight * cross_entropy, axis=1)
    return focal_loss
