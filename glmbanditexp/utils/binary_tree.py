import numpy as np
import math

class BinaryTreeMechanism:
    """
    Binary-tree mechanism for private prefix sums of d×d symmetric matrices.

    Privacy accounting (zCDP/Gaussian-DP):
      - Target overall (epsilon, delta).
      - Let m = ceil(log2(T)) + 1 be the max #nodes a single prefix needs.
      - Convert (epsilon, delta) -> total zCDP budget rho_tot via:
            epsilon = rho_tot + 2 * sqrt(rho_tot * log(1/delta))
        => with a = sqrt(log(1/delta)) and t = sqrt(rho_tot),
            t = max(0, -a + sqrt(a*a + epsilon)),  rho_tot = t^2
      - Split evenly across nodes: rho_node = rho_tot / m (additive in zCDP).
      - Gaussian mech with l2-sensitivity C has rho = C^2 / (2 sigma^2),
        so per-node sigma is:
            sigma_node = C / sqrt(2 * rho_node)

    IMPORTANT:
      - We sample ONE Gaussian noise matrix per node in __init__ and REUSE it
        for every query. This is crucial for obtaining the √log(T) scaling.
    """

    def __init__(self, T, epsilon, delta, d, C=1.0, lam=1e-6, psd_floor=1e-12, rng=None):
        assert T >= 1
        self.T = int(T)
        self.height = math.ceil(math.log2(T)) + 1
        self.tree_size = 2 ** self.height

        # noiseless partial sums per node
        self.d = int(d)
        self.tree = [np.zeros((self.d, self.d)) for _ in range(self.tree_size)]
        self.count = 0

        # privacy / regularization
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.C = float(C)
        self.lam = float(lam)
        self.psd_floor = float(psd_floor)

        # compute per-node sigma via zCDP (tight & clean)
        self.m = math.ceil(math.log2(self.T)) + 1
        self.sigma_node = self._compute_sigma_node_zcdp()

        # pre-sample ONE symmetric Gaussian noise matrix per node
        self.noise = [np.zeros((self.d, self.d)) for _ in range(self.tree_size)]
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()
        for idx in range(1, self.tree_size):
            z = self.rng.normal(0.0, self.sigma_node, size=(self.d, self.d))
            z_sym = 0.5 * (z + z.T)  # symmetric
            self.noise[idx] = z_sym

    # ---------- privacy math ----------

    def _compute_sigma_node_zcdp(self):
        """Compute per-node Gaussian std using zCDP accounting."""
        # total (eps, delta) -> rho_tot
        a = math.sqrt(max(1e-300, math.log(1.0 / self.delta)))  # guard tiny deltas
        t = max(0.0, math.sqrt(a * a + self.epsilon) - a)       # t = sqrt(rho_tot)
        rho_tot = t * t
        if rho_tot <= 0.0:
            # extremely tiny epsilon; fall back to huge sigma
            return float('inf')

        rho_node = rho_tot / self.m
        sigma_node = self.C / math.sqrt(2.0 * rho_node)
        return sigma_node

    # ---------- numerics ----------

    def _add_regularization(self, M):
        # add λI to ensure uniform regularization across queries
        return M + self.lam * np.eye(self.d)

    def _project_to_psd(self, M):
        # symmetric-ize then clip eigenvalues
        M = 0.5 * (M + M.T)
        vals, vecs = np.linalg.eigh(M)
        vals_clipped = np.maximum(vals, self.psd_floor)
        return (vecs * vals_clipped) @ vecs.T

    # ---------- tree ops ----------

    def _accumulate_upwards(self, index, value):
        """Add 'value' to node 'index' and all its ancestors (noiseless)."""
        while index > 0:
            self.tree[index] += value
            index //= 2

    def update(self, x):
        """
        Insert a new d×d (symmetric) value x at the next leaf.
        """
        self.count += 1
        if self.count > self.T:
            raise ValueError("Exceeded declared T updates.")
        # index of leaf: leaves start at 2^(height-1)
        leaf_index = self.count + (2 ** (self.height - 1)) - 1
        self._accumulate_upwards(leaf_index, x)

    def _prefix_cover_nodes(self):
        """Return a minimal set of tree nodes covering prefix [1..count]."""
        if self.count == 0:
            return []

        base = 2 ** (self.height - 1)
        left = base
        right = base + self.count - 1
        nodes = []

        # Invariant: [left, right] (inclusive) is exactly the uncovered remainder
        # of the original prefix leaf interval, traversed left-to-right.
        while left <= right:
            if left % 2 == 1:
                nodes.append(left)
                left += 1
            if right % 2 == 0:
                nodes.append(right)
                right -= 1
            left //= 2
            right //= 2

        return nodes

    def query(self):
        """
        Return DP prefix sum up to current count.

        Implementation detail:
          We traverse from the current leaf to the root; whenever the
          current node is a RIGHT child, we add its LEFT sibling (disjoint
          segment) to the result — both the stored noiseless sum AND its
          fixed per-node noise. Finally, we also add the current leaf node
          content and its noise.
        """
        if self.count == 0:
            result = self._add_regularization(np.zeros((self.d, self.d)))
            return self._project_to_psd(result)

        result = np.zeros((self.d, self.d))

        for node in self._prefix_cover_nodes():
            result += self.tree[node] + self.noise[node]

        # add uniform regularization and project to PSD for numerical safety
        result = self._add_regularization(result)
        result = self._project_to_psd(result)
        return result
