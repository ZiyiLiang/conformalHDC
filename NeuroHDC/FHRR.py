### FHRR Representation
#choose Multivariate Gaussian $N_n(0,\Sigma)$ as the distribution to generate the encoding codebook for n features.
import numpy as np 
 

class RFF:
    def __init__(self, dimension, n_feature, seed = 123):
        """
        Params:
            dimension: The dimension of the hypervectors. (8k-20k)
            n_feature: number of features in data
        """
        self.n_feature = n_feature
        self.D = dimension
        self.rng = np.random.default_rng(seed=seed)

    def gen_basis(self,cov=None):
        ''' 
            Generate basis using multivariate normal distribution
        Params: 
            cov(n,n): covariance matrix for hypervectors
        '''

        n = self.n_feature 
        if n>1:
            mean = np.zeros((n,)) 
            W = self.rng.multivariate_normal(mean, cov, (self.D)) # shape(D,n)
        if n==1:
            W = self.rng.normal(0, cov, (self.D,1))
        return W
    
    def gen_time_base(self):
        theta = self.rng.uniform(0, 2*np.pi, size=(self.D,))
        return np.exp(1j * theta)
 
    def encode_trial(self, W_neurons, X_trial, Time_Base, beta = 0.2):
        """
        X_trial: (T,N)   
        W_neurons:(D,N)
        time_codebook: shape (T, D)
        beta: 
            beta: fractional power
        return: trial hypervector shape (D,)
        """

        # (D, N) @ (N, T) -> (D, T)
        projections = W_neurons @ X_trial.T

        # 2. Exponentiate
        hv_neurons = np.exp(1j * beta * projections) # (D, T)

 
        hv_sum = np.zeros(self.D, dtype=np.complex128)
        
        for ti in range(X_trial.shape[0]): 

            # bind the time with the neuron features
            hv_timed = self.bind(hv_neurons[:,ti], self.permute(Time_Base,shift=ti))  # bind with time tag
            
            # bundle over time bins
            hv_sum += hv_timed

        return self.normalize(hv_sum)

    def encode_all(self, W_neurons, X, Time_Base,beta):
        # X: (trials, timebins, neurons)
        trials = X.shape[0]
        H = np.empty((trials, self.D), dtype=np.complex128)
        for i in range(trials): 
            H[i] = self.encode_trial(W_neurons, X[i], Time_Base,beta)
        return H

    def normalize(self, hv, eps=1e-12):
            mag = np.abs(hv)
            return hv / np.maximum(mag, eps)
    
    def bundle(self, hv1, hv2):
        """Performs element-wise addition (bundling) followed by binarization."""
        return self.normalize(hv1 + hv2)
    
    def bind(self, hv1, hv2):
        """Performs element-wise multiplication (binding) between two hypervectors."""
        return self.normalize(hv1 * hv2)
    def permute(self, hv, shift=1):
            """Performs permutation (shift to the right by default)."""
            return np.roll(hv, shift)
    def similarity(self, hv1, hv2):
        """
        Computes cosine similarity between two hypervectors.
        Normalized dot product.
        
        :param hv1: First hypervector.
        :param hv2: Second hypervector.
        :return: Cosine similarity value.
        """
        hv1 = np.ravel(hv1)  # ensures shape (D,)
        hv2 = np.ravel(hv2)  # ensures shape (D,)
        delta = np.dot(hv1,np.conj(hv2)).real/self.D
        return delta
    
    def build_class_prototypes(self,trian_hvs,y, weights = None) :
        '''
            We bundle all sample HVs per class and sign to get one prototype per class.
        Params:
            train_hvsL: (n_trial,D)
            y: (n_trial,) class labels
            weights: class weight
        ''' 

        label = np.unique(y)
        if weights is None:
            weights = np.ones(trian_hvs.shape[0])
        codebook_1pass = {}

        for il in label:
            il_idx = np.where(y == il)[0]
            
            # 1. Calculate the SUM of all hypervectors for this class. 
            hv_il_sum = np.sum(trian_hvs[il_idx] * weights[il_idx].reshape(-1,1) , axis=0) 
            
            # 2. Normalize the average hypervector to get the final, stable class prototype.
            codebook_1pass[il] = self.normalize(hv_il_sum)

        return codebook_1pass
    
    def train_iterative_novelty(self, train_hvs, y, iterations=3):
        """
        Iteratively updates prototypes by weighting samples based on 
        how noval they are from the current prototype.
        
        Formula: Weight = 1 - Similarity(Prototype, Sample)
        """
        # Step 1: Initial Pass (Uniform weights)
        weights = np.ones(train_hvs.shape[0])
        codebook = self.build_class_prototypes(train_hvs, y, weights)
        
        for it in range(iterations): 
            # Update weights for every sample
            for i in range(train_hvs.shape[0]):
                label = y[i]
                current_prototype = codebook[label]
                sample_hv = train_hvs[i]
                
                # Calculate Similarity  
                sim = self.similarity(sample_hv, current_prototype)
                
                # NEW WEIGHT: Higher weight if similarity is LOW 
                weights[i] = max(1.0 - sim, 0.01)

            # Re-build prototypes with the new "Novelty" weights
            codebook = self.build_class_prototypes(train_hvs, y, weights)
            
        return codebook
    def decode_1d(self,hv,codebook):
        pred = max(codebook, key=lambda k: self.similarity(hv, codebook[k]))
        return pred
    
    def decode(self,hv,codebook): 
        # hv: (batch,dimension)
        y_test_pred = []
        for i in range(hv.shape[0]):
            pred = self.decode_1d(hv[i,:], codebook)
            y_test_pred.append(pred)
        return y_test_pred
 
    # -----------------------------
    #  iterative learning: reinforce correct + relearn incorrect
    # -----------------------------

    def iterate_learn(
            self,
            encoded_x_train, encoded_x_val, 
            y_train, y_val, 
            codebook_1pass,
            initial_alpha=1e-2,   
            decay_rate=1e-4,
            step_size=20,
            patience=5):
        best_val = -np.inf
        no_improve = 0
        history = [] 

        for i in range(encoded_x_val.shape[0]): 
            # Decaying Alpha (Learning Rate)
            alpha = initial_alpha / (1 + decay_rate * i)  
            # decode on my query
            query = encoded_x_val[i,:]
            pred_y = self.decode_1d(query, codebook_1pass)
            true_y = y_val[i] 

            # 1. FORGET (if wrong): Softly pull the wrong codebook vector away
            if pred_y != true_y:
                # print('peanlize')
                #C_wrong -= α query
                codebook_1pass[pred_y] =  codebook_1pass[pred_y] - alpha * query

                codebook_1pass[true_y] += alpha * query # Move purely to perceptron style?

                # Re-normalize to keep phasors valid
                codebook_1pass[pred_y] = self.normalize(codebook_1pass[pred_y])
                codebook_1pass[true_y] = self.normalize(codebook_1pass[true_y])
           
            # eval after X steps
            if (i + 1) % step_size == 0:
                # eval the train
                y_train_pred = self.decode(encoded_x_train,codebook_1pass)
                # eval the validation
                y_val_pred = self.decode(encoded_x_val,codebook_1pass)

                train_acc= np.mean(y_train_pred==y_train) 
                val_acc= np.mean(y_val_pred==y_val) 

                history.append((i + 1, train_acc, val_acc))
                print(f"STEP {i+1} | train {train_acc:.3f} | val {val_acc:.3f}")

                if val_acc > best_val + 1e-12:
                    best_val = val_acc
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stop: no val improvement for {patience} evals.")
                        break

        return codebook_1pass