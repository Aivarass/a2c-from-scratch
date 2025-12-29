import java.util.Arrays;
import java.util.Random;

/**
 * Tiny shallow Actor-Critic neural net:
 *   inputs -> tanh hidden -> (actor logits for each action) + (critic value)
 *
 * Update rules (one-step TD(0)):
 *   delta = r + gamma * V(s') - V(s)          (or r - V(s) if terminal)
 *
 * Critic:  w_c += alphaCritic * delta * grad V(s)
 * Actor:   w_a += alphaActor  * delta * grad log pi(a|s)
 *
 * Actor grad for softmax logits:
 *   d log pi(a) / d logit(k) = (I[k==a] - pi(k))
 *
 * We apply that to actor head weights, and backprop through the trunk.
 */
public class TinyActorCriticANN {

    private final double ENTROPY_BETA = 0.005;     // entropy bonus coefficient

    private final double ENTROPY_EPS  = 1e-12;    // safety for log(0)

    private final int inputDim;
    private final int hiddenUnits;
    private final int actionCount;

    // Trunk: hidden = tanh( Wih * x + bh )
    private final double[][] wInputHidden; // [H][D]
    private final double[] bHidden;        // [H]

    // Actor head: logits = Wa * hidden + ba
    private final double[][] wHiddenActor; // [A][H]
    private final double[] bActor;         // [A]

    // Critic head: value = Wv * hidden + bv
    private final double[] wHiddenValue;   // [H]

    private double bValueMutable;          // keep mutable bias separate

    private final Random rng;

    private final double[] wEnt;           // scratch buffer for entropy gradient

    // ---- Scratch buffers (avoid allocations each step) ----
    private final double[] hidden;     // [H]
    private final double[] logits;     // [A]
    private final double[] probs;      // [A]

    public TinyActorCriticANN(int inputDim, int hiddenUnits, int actionCount, long seed) {
        if (inputDim <= 0 || hiddenUnits <= 0 || actionCount <= 1) {
            throw new IllegalArgumentException("inputDim>0, hiddenUnits>0, actionCount>1 required");
        }
        this.inputDim = inputDim;
        this.hiddenUnits = hiddenUnits;
        this.actionCount = actionCount;
        this.rng = new Random(seed);

        this.wInputHidden = new double[hiddenUnits][inputDim];
        this.bHidden = new double[hiddenUnits];

        this.wHiddenActor = new double[actionCount][hiddenUnits];
        this.bActor = new double[actionCount];

        this.wHiddenValue = new double[hiddenUnits];
        this.bValueMutable = 0.0;

        this.hidden = new double[hiddenUnits];
        this.logits = new double[actionCount];
        this.probs = new double[actionCount];

        this.wEnt = new double[actionCount];

        initWeightsXavier();
    }

    public TinyActorCriticANN(int inputDim, int hiddenUnits, int actionCount) {
        this(inputDim, hiddenUnits, actionCount, System.nanoTime());
    }

    private void initWeightsXavier() {
        // Xavier-ish init for tanh trunk
        double limitIH = Math.sqrt(6.0 / (inputDim + hiddenUnits));
        for (int h = 0; h < hiddenUnits; h++) {
            for (int d = 0; d < inputDim; d++) {
                wInputHidden[h][d] = uniform(-limitIH, limitIH);
            }
            bHidden[h] = 0.0;
        }

        // Actor head
        double limitHA = Math.sqrt(6.0 / (hiddenUnits + actionCount));
        for (int a = 0; a < actionCount; a++) {
            for (int h = 0; h < hiddenUnits; h++) {
                wHiddenActor[a][h] = uniform(-limitHA, limitHA);
            }
            bActor[a] = 0.0;
        }

        // Critic head
        double limitHV = Math.sqrt(6.0 / (hiddenUnits + 1.0));
        for (int h = 0; h < hiddenUnits; h++) {
            wHiddenValue[h] = uniform(-limitHV, limitHV);
        }
        bValueMutable = 0.0;
    }

    private double uniform(double min, double max) {
        return min + (max - min) * rng.nextDouble();
    }

    // -----------------------
    // Public API (inference)
    // -----------------------

    /** Returns a copy of the current policy probabilities pi(.|s). */
    public double[] policyProbs(double[] x) {
        forward(x);
        return Arrays.copyOf(probs, probs.length);
    }

    /** Returns pi(action|s) only (efficient if you just need one). */
    public double policyProb(double[] x, int action) {
        forward(x);
        return probs[action];
    }

    /** Samples an action according to the current softmax policy. */
    public int sampleAction(double[] x) {
        forward(x);
        double r = rng.nextDouble();
        double cdf = 0.0;
        for (int a = 0; a < actionCount; a++) {
            cdf += probs[a];
            if (r <= cdf) return a;
        }
        return actionCount - 1; // numerical safety
    }

    /** Returns V(s). */
    public double value(double[] x) {
        forwardHiddenOnly(x);
        return computeValueFromHidden();
    }

    // -----------------------
    // Learning / update
    // -----------------------

    /**
     * One-step Actor-Critic update.
     *
     * @param s           current state features
     * @param action      action taken at s
     * @param reward      reward observed after action
     * @param sNext       next state features (ignored if terminal)
     * @param terminal    whether episode ended after this transition
     * @param alphaActor  actor step size
     * @param alphaCritic critic step size
     * @param alphaTrunk  trunk (shared hidden layer) step size
     * @param gamma       discount factor
     * @return delta (TD error), useful for logging/debug
     */
    public double update(double[] s,
                         int action,
                         double reward,
                         double[] sNext,
                         boolean terminal,
                         double alphaActor,
                         double alphaCritic,
                         double alphaTrunk,
                         double gamma) {

        // ---- Forward on current state: get hidden, probs, and V(s) ----
        forward(s);
        double vS = computeValueFromHidden();

        // Snapshot for critic and actor weights
        double[] wVsnap = Arrays.copyOf(wHiddenValue, hiddenUnits);
        double[][] wAsnap = new double[actionCount][hiddenUnits];
        for (int k = 0; k < actionCount; k++) {
            System.arraycopy(wHiddenActor[k], 0, wAsnap[k], 0, hiddenUnits);
        }

        // ---- Bootstrap V(s') if non-terminal ----
        double vNext = 0.0;
        if (!terminal) {
            forwardHiddenOnly(sNext);
            vNext = computeValueFromHidden();
        }

        double delta = reward + gamma * vNext - vS;
        double deltaClipped = clip(delta, -5.0, 5.0);  // prevent large updates

        // We need trunk gradients; capture snapshots before modifying trunk:
        // - For critic trunk gradient we need wHiddenValue
        // - For actor trunk gradient we need wHiddenActor[*][h] and probs
        // We'll backprop in two parts: critic contribution + actor contribution.

        // ---------------- Critic update (value head) ----------------
        // V = sum_h wV[h]*hidden[h] + bV
        // dV/dwV = hidden, dV/dbV = 1
        for (int h = 0; h < hiddenUnits; h++) {
            wHiddenValue[h] += alphaCritic * deltaClipped * hidden[h];
        }
        bValueMutable += alphaCritic * deltaClipped;

        // Precompute entropy “weights”: w_j = p_j * (log p_j + 1)
        double sumW = 0.0;
        for (int j = 0; j < actionCount; j++) {
            wEnt[j] = probs[j] * (Math.log(probs[j] + ENTROPY_EPS) + 1.0);
            sumW += wEnt[j];
        }

        for (int k = 0; k < actionCount; k++) {
            double pg = ((k == action) ? 1.0 : 0.0) - probs[k];

            double dH_dlogit = (-wEnt[k]) + (probs[k] * sumW);

            double g = (deltaClipped * pg) + (ENTROPY_BETA * dH_dlogit);

            double step = alphaActor * g;

            for (int h = 0; h < hiddenUnits; h++) {
                wHiddenActor[k][h] += step * hidden[h];
            }
            bActor[k] += step;
        }

        // ---------------- Backprop into trunk (critic + actor) ----------------
        // For each hidden unit h:
        //   z = bH + Wih*x
        //   hidden = tanh(z)
        //   dh/dz = 1 - hidden^2
        //
        // Critic part: dJ/dh = alphaCritic * delta * wV[h]
        //
        // Actor part:
        //   log pi(action) gradient wrt hidden:
        //     d log pi(action)/dh = sum_k (I[k==a]-pi[k]) * Wa[k][h]
        //   so actor trunk contribution:
        //     alphaActor * delta * [that] * dh/dz
        //
        // Total trunk gradient:
        //   chain = (criticPart + actorPart) * (1 - hidden^2)
        //   Wih[h][d] += chain * x[d]
        //   bH[h]     += chain
        //
        for (int h = 0; h < hiddenUnits; h++) {
            double dh_dz = 1.0 - hidden[h] * hidden[h];

            double criticPart = deltaClipped * wVsnap[h];

            double actorSum = 0.0;
            for (int k = 0; k < actionCount; k++) {
                double g = ((k == action) ? 1.0 : 0.0) - probs[k];
                actorSum += g * wAsnap[k][h];
            }
            double actorPart  = deltaClipped * actorSum;

            double chain = alphaTrunk * (criticPart + actorPart) * dh_dz;

            for (int d = 0; d < inputDim; d++) {
                wInputHidden[h][d] += chain * s[d];
            }
            bHidden[h] += chain;
        }

        return delta;
    }

    // -----------------------
    // Forward helpers
    // -----------------------

    /** Full forward pass: hidden -> logits -> probs. */
    private void forward(double[] x) {
        forwardHiddenOnly(x);

        // logits
        for (int a = 0; a < actionCount; a++) {
            double z = bActor[a];
            for (int h = 0; h < hiddenUnits; h++) {
                z += wHiddenActor[a][h] * hidden[h];
            }
            logits[a] = z;
        }

        softmaxInPlace(logits, probs);
    }

    /** Only compute hidden activations (used for V(s) bootstrap). */
    private void forwardHiddenOnly(double[] x) {
        for (int h = 0; h < hiddenUnits; h++) {
            double z = bHidden[h];
            for (int d = 0; d < inputDim; d++) {
                z += wInputHidden[h][d] * x[d];
            }
            hidden[h] = Math.tanh(z);
        }
    }

    private double computeValueFromHidden() {
        double v = bValueMutable;
        for (int h = 0; h < hiddenUnits; h++) {
            v += wHiddenValue[h] * hidden[h];
        }
        return v;
    }

    private static void softmaxInPlace(double[] inLogits, double[] outProbs) {
        double max = inLogits[0];
        for (int i = 1; i < inLogits.length; i++) max = Math.max(max, inLogits[i]);

        double sum = 0.0;
        for (int i = 0; i < inLogits.length; i++) {
            double e = Math.exp(inLogits[i] - max);
            outProbs[i] = e;
            sum += e;
        }
        double inv = 1.0 / sum;
        for (int i = 0; i < outProbs.length; i++) outProbs[i] *= inv;
    }

    private static double clip(double x, double lo, double hi) {
        return Math.max(lo, Math.min(hi, x));
    }
}