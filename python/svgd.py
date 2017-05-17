import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist

class SVGD():

    def __init__(self):
        pass

    # Returns control functional weights
    def getWeights(self, KpMat):
        condNumber = self.getConditionNumber(KpMat)
        z = KpMat.shape[0]

        # Get weights
        KPrime = KpMat + condNumber * z * np.identity(z)
        num = np.matmul(np.ones(z),np.linalg.inv(KPrime))
        denom = 1 + np.matmul(num,np.ones(z))
        weights = num / denom

        weights = weights / sum(weights)

        return (weights)

    # Given a kernel matrix K, let lambda be smallest power of 10 such that
    # kernel matrix K0 + lamba*I has condition number lower than 10^10
    # Note we use 2-norm for computing condition number
    def getConditionNumber(self, K):
        condNumber = 10e-10
        condA = 10e11
        matSize = K.shape[0]
        while condA > 10e10:
            condNumber = condNumber * 10
            A = K + condNumber * np.identity(matSize)
            condA = np.linalg.norm(A, ord=2) * np.linalg.norm(np.linalg.inv(A), ord=2)
        return (condNumber)

    def svgd_kernel(self, theta, lnprob, h = -1):
        n,d = theta.shape
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(n+1))

        # compute the rbf kernel
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(d):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)

        lnpgrad = lnprob(theta)
        grad_theta = (np.matmul(Kxy, lnpgrad) + dxkxy) / n

        return grad_theta

    # Compute gradient update for theta using svgd random subset (with optional control functional)
    def svgd_kernel_subset(self, theta, lnprob, h=-1, m=-1, cf = False):
        if m == -1:
            m = int(theta.shape[0]/5);

        n,d = theta.shape
        yInd = np.random.choice(n, m, replace=False)
        y = theta[yInd]

        pairwise_dists = cdist(theta, y)**2

        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(n+1))

        # compute the rbf kernel
        Kxy = np.exp( -pairwise_dists / h**2 / 2)
        Sqy = lnprob(y)
        if cf == True : # Using control functional

            sqxdy_part = np.array([np.sum(np.multiply(Sqy,y),axis=1),]*m).T
            sqxdy = -(np.matmul(Sqy,y.T)- sqxdy_part)/ h**2
            dxsqy = sqxdy.T
            dxdy = -pairwise_dists[yInd]/h**4 +d/h**2
            KxySub = Kxy[yInd]

            KpMat = (np.matmul(Sqy, Sqy.T) + sqxdy + dxsqy + dxdy)
            KpMat = np.multiply(KpMat, KxySub)

            weights = self.getWeights(KpMat)
            Kxy = np.multiply(Kxy, np.matlib.repmat(weights, n, 1))

        dxkxy = -np.matmul(Kxy, y)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(d):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)


        grad_theta = (np.matmul(Kxy, Sqy) + dxkxy)
        if cf == False:
            grad_theta = grad_theta / m
        return grad_theta

    # Perform a step of adam update
    def get_adamUpdate(self, iterInd, ori_grad, hist_grad, stepsize = 1e-3, alpha = 0.9, fudge_factor = 1e-6):
        if iterInd == 0:
            hist_grad = hist_grad + ori_grad ** 2
        else:
            hist_grad = alpha * hist_grad + (1 - alpha) * (ori_grad ** 2)

        adj_grad = np.divide(ori_grad, fudge_factor+np.sqrt(hist_grad))

        return (stepsize * adj_grad, hist_grad)

    # Compute gradient update for y
    def svgd_kernel_grady(self, theta, y, Sqx, h=-1, uStat=True, regCoeff=0.1):
        n,d = theta.shape
        m = y.shape[0]

        pairwise_dists = cdist(theta, y)**2

        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(n+1))

        # compute the rbf kernel
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        yGrad = np.zeros((m,d));

        # Compute gradient
        for yInd in range(m):
            Kxy_cur = Kxy[:,yInd];
            xmy = (theta - np.tile(y[yInd,:],[n,1]))/h**2
            Sqxxmy = Sqx - xmy;
            back = np.tile(np.array([Kxy_cur]).T,(1,d)) * Sqxxmy
            inner = np.tile(np.array([np.sum(np.matmul(back, back.T),axis=1)]).T,[1,d])
            yGrad[yInd,:] = np.sum(xmy * inner,axis=0) + np.sum(back,axis=0) * np.sum(Kxy_cur)/h**2

            # For U-statistic
            if uStat:
                front_u = np.tile(np.array([(Kxy_cur**2) * np.sum(Sqxxmy **2,axis=1)]).T,[1,d]) * xmy;
                back_u = np.tile(np.array([Kxy_cur**2 / h**2]).T,[1,d]) * Sqxxmy

                yGrad[yInd,:] = yGrad[yInd,:] - np.sum(front_u + back_u,axis=0)

        if uStat:
            yGrad = yGrad * 2 / (n*(n-1)*m);
        else:
            yGrad = yGrad * 2 / (n**2 * m);

        if regCoeff > 0 :
            H_y = cdist(y, y)**2
            Kxy_y = np.exp( -H_y / h**2 / 2)
            sumKxy_y = np.sum(Kxy_y,axis=1)
            yReg = (y * np.tile(np.array([sumKxy_y]).T,[1,d]) - np.matmul(Kxy_y,y))/(h**2 * m)

        yGrad = yGrad + regCoeff * yReg
        return (yGrad)

    # Compute gradient update for h
    def svgd_kernel_gradh(self, theta, y, Sqx, h=-1, uStat=True):
        n,d = theta.shape
        m = y.shape[0]

        H = cdist(theta, y)**2

        if h < 0: # if h < 0, using median trick
            h = np.median(H)
            h = np.sqrt(0.5 * h / np.log(n+1))

        # compute the rbf kernel
        Kxy = np.exp( -H / h**2 / 2)

        hGrad = 0;

        # For each induced point
        for yInd in range(m):
            Kxy_cur = Kxy[:,yInd]
            H_cur = H[:,yInd]
            xmy = (theta - np.tile(y[yInd,:],[n,1]))/h**2
            Sqxxmy = Sqx - xmy

            part2 = np.tile(np.array([Kxy_cur]).T,[1,d]) * Sqxxmy
            part1_1 = np.tile(np.array([H_cur/h**3]).T,[1,d]) * part2
            part1_2 = np.tile(np.array([Kxy_cur]).T,[1,d]) * (2*xmy / h**3)
            part = np.matmul(part1_1 + part1_2, part2.T)
            hGrad = hGrad + np.sum(np.sum(part,axis=1))

            if uStat:
                front_u = (Kxy_cur**2) * (H_cur/h**3) * np.sum(Sqxxmy**2, axis=1)
                back_u = np.sum((2*xmy/h**3) * Sqxxmy,axis=1)
                hGrad = hGrad - np.sum(Kxy_cur**2 * (front_u + back_u),axis=0)

        if uStat:
            hGrad = hGrad * 2 / (n*(n-1)*m);
        else:
            hGrad = hGrad * 2 / (n**2 * m);

        return (hGrad)

    def svgd_kernel_inducedPoints(self, theta, lnprob, h=-1, m=-1, adver = False, adverMaxIter = 5, stepsize = 1e-3, alpha = 0.9):
        if m == -1:
            m = int(theta.shape[0]/5);

        n,d = theta.shape
        yInd = np.random.choice(n, m, replace=False)
        y = theta[yInd]

        Sqx = lnprob(theta);

        # If we want to perform EM
        if adver == True:
            # Perform update emMaxIter number of times
            fudge_factor = 1e-6

            for adverIter in range(0,adverMaxIter):
                grad_y = self.svgd_kernel_grady(theta, y, Sqx, h=h)

                [update_y,hist_grad] = self.get_adamUpdate(adverIter, grad_y, self.y_historical_grad,stepsize = stepsize, alpha = alpha)
                y = y + update_y
                self.y_historical_grad = hist_grad

                grad_h = self.svgd_kernel_gradh(theta, y, Sqx, h=h)
                [update_h, hist_grad] = self.get_adamUpdate(adverIter, grad_h, self.h_historical_grad,stepsize = stepsize, alpha = alpha)
                h = h + update_h
                self.h_historical_grad = hist_grad

        pairwise_dists = cdist(theta, y)**2
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(n+1))

        # compute the rbf kernel
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        innerTerm_1 = np.matmul(Kxy.T, (Sqx - theta/ h**2))
        sumkxy = np.sum(Kxy, axis=0)
        innerTerm_2 = np.multiply(np.tile(np.array([sumkxy]).T,(1,d)), y/h**2)
        innerTerm = (innerTerm_1 + innerTerm_2)/n

        gradTheta = np.matmul(Kxy, innerTerm)/m
        return (gradTheta)


    def update(self, x0, lnprob, n_iter = 1000, stepsize = 1e-3, bandwidth = -1,
        alpha = 0.9, method='none',m=-1,cf=False, adver=False, adverMaxIter = 5, debug = False, regCoeff = 0.1):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')

        theta = np.copy(x0)

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        self.y_historical_grad = 0
        self.h_historical_grad = 0
        for iter in range(n_iter):
            if debug and (iter+1) % 1000 == 0:
                print ('iter ' + str(iter+1))

            # calculating the kernel matrix
            if method == 'subparticles':
                grad = self.svgd_kernel_subset(theta, lnprob, h=-1, m = m, cf = cf)
            elif method == 'inducedPoints':
                grad = self.svgd_kernel_inducedPoints(theta,lnprob,h=-1,m=m, adver=adver, adverMaxIter = adverMaxIter)
            elif method == 'none':
                grad = self.svgd_kernel(theta,lnprob, h=-1)

            [adam_grad, historical_grad] = self.get_adamUpdate(iter, grad, historical_grad, stepsize, alpha, fudge_factor)
            theta = theta + adam_grad

        return theta
