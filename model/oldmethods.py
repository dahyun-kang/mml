    def forward_classwiseknn(self, x):
        out = self.backbone(x)

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)
            # B, C, N -> B, C, K
            topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=False)
            # 1, C, 1
            class_idx = torch.arange(self.num_classes).unsqueeze(0).unsqueeze(-1)
            # C, N, D [[1, C, 1], [B, C, K]] -> B, C, K, D
            knnemb = self.memory_list[class_idx, indices]
            # B, C, K, D -> B, C, D
            knnemb_avg = torch.mean(knnemb, dim=2)
            # (B, 1, D), (B, C, D) -> B, (1 + C), D
            context_emb = torch.cat([out.unsqueeze(1), knnemb_avg], dim=1)

        tr_q = out.unsqueeze(1)
        out = self.knnformer(tr_q, context_emb, context_emb)

        sim = torch.einsum('b d, c d -> b c', out[:, 0], self.global_proto)

        return F.log_softmax(out, dim=1)

    def forward_nakata22(self, x):
        out = self.backbone(x)

        def majority_vote(input):
            stack = []
            count = [0]*self.num_classes
            for item in input:
                count[item.cpu().item()] += 1
                if not stack or stack[-1] == item:
                    stack.append(item)
                else:
                    stack.pop()

            # onehot = (input[0] if not stack else stack[0]).cpu().item() # real majority vote
            onehot = torch.argmax(torch.tensor(count)).item() # just vote
            result = torch.tensor([0.]*self.num_classes)
            result[onehot] = 1.0

            return result.to(self.device)

        with torch.no_grad():
            num_samples = self.memory_list.shape[1]
            all_features = self.memory_list.view([-1, self.memory_list.shape[2]])

            similarity_mat = torch.einsum('b d, n d -> b n', F.normalize(out, dim=-1), F.normalize(all_features, dim=-1))

            topk_sim, indices = similarity_mat.topk(k=self.args.k, dim=-1, largest=True, sorted=False)

            indices = torch.div(indices, num_samples, rounding_mode='trunc')
            voting_result = torch.stack(list(map(majority_vote, indices)))

        return voting_result

    def forward_prototypematching(self, x):
        out = self.backbone(x)
        tr_q = out.unsqueeze(1)

        out = self.knnformer(tr_q, tr_q, tr_q)
        out = torch.einsum('b d, c d -> b c', out[:, 0], self.global_proto)

        return F.log_softmax(out, dim=1)

    def forward_naiveaddedformer(self, x):
        # this model flattens the output without pooling
        # B (D H W)
        out = self.backbone(x)

        out = rearrange(out, 'b (d l) -> b l d', l=49, d=self.dim)  # TODO: remove hardcode
        out = self.knnformer(out)
        out = self.fc(out.mean(dim=1))
        return F.log_softmax(out, dim=1)

    def forward_directknnmatching(self, x):
        out = self.backbone(x)

        '''
        classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)

        classwise_sim = self.memory_list(out)
        classwise_sim = rearrange(classwise_sim, 'b (c n) -> b c n', c=self.num_classes)

        # B, C, N -> B, C, K
        topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=False)
        '''

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)

            # B, C, N -> B, C, K
            topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=False)

            # 1, C, 1
            class_idx = torch.arange(self.num_classes).unsqueeze(0).unsqueeze(-1)

            # C, N, D [[1, C, 1], [B, C, K]] -> B, C, K, D
            knnemb = self.memory_list[class_idx, indices]

        topk_sim = torch.einsum('b d, b c k d -> b c k', out, knnemb)
        # B, C, K -> B, C
        topk_sim = topk_sim.mean(dim=-1)

        return F.log_softmax(topk_sim, dim=1)

    def forward_m18_prototypeupdate_generictoken(self, x):
        """
        0301 m18
        - unnormalized generic tokens as input
        - standardized prototypes after update
        - l2normalized features used
        """

        out = self.backbone(x)
        # normalize
        out = F.normalize(out, p=2, dim=-1)
        batchsize = out.shape[0]

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)
            if self.training:  # to ignore self-voting
                # B, C, N -> B, C, K
                topk_sim, indices = classwise_sim.topk(k=self.args.k + 1, dim=-1, largest=True, sorted=True)
                # indices = indices[:, :, 1:]
                top1_indices = indices[:, :, 0]
                max_class_indices = top1_indices.argmax(dim=1)  # highly likely the self (or another twin in the feature space)
                indices[range(batchsize), max_class_indices, :-1] = indices[range(batchsize), max_class_indices, 1:]
                indices = indices[:, :, :-1]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)

            # 1, C, 1
            class_idx = torch.arange(self.num_classes).unsqueeze(0).unsqueeze(-1)
            # C, N, D [[1, C, 1], [B, C, K]] -> B, C, K, D
            knnemb = self.memory_list[class_idx, indices]

            knnemb = torch.cat([repeat(self.global_proto, 'c d -> b c 1 d', b=batchsize), knnemb], dim=2)
            # knnemb = rearrange(knnemb, 'b c k d -> (b c) k d')
            knnemb = rearrange(knnemb, 'b c k d -> b (c k) d')

        # (M, D), (B, CK, D) -> B, M, D
        updated_tokens = self.knnformer(repeat(self.generic_tokens, 'm d -> b m d', b=batchsize), knnemb, knnemb)
        # updated_tokens = F.normalize(updated_tokens, p=2, eps=1e-6, dim=-1)

        # no kNN baseline; no prototype update at all!
        # gtokens = self.generic_tokens.unsqueeze(0)
        # _updated_proto = self.knnformer2(self.global_proto.unsqueeze(0), gtokens, gtokens)

        # (B, C, D), (B, M, D) -> B, C, D
        _updated_proto = self.knnformer2(repeat(self.global_proto, 'c d -> b c d', b=batchsize), updated_tokens, updated_tokens)

        # standization & l2 normalization
        _updated_proto = self.standardize(_updated_proto)

        # output becomes NaN if it's commented!!
        updated_proto = F.normalize(_updated_proto, p=2, eps=1e-6, dim=-1)

        sim = torch.einsum('b d, b c d -> b c', out, updated_proto)

        # no kNN baseline; no prototype update at all!
        # sim = torch.einsum('b d, c d -> b c', out, updated_proto.squeeze(0))
        return F.log_softmax(sim / 0.01, dim=1)

    def forward_old_m151617(self, x):
        idx = idx % self.dm.max_num_samples
        out = self.backbone(x)
        # normalize
        out = F.normalize(out, p=2, dim=-1)
        batchsize = out.shape[0]

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, c n d -> b c n', out, self.memory_list)
            '''
            if self.training:  # to ignore self-voting
                # B, C, N -> B, C, K
                topk_sim, indices = classwise_sim.topk(k=self.args.k + 1, dim=-1, largest=True, sorted=True)
                # indices = indices[:, :, 1:]
                top1_indices = indices[:, :, 0]
                max_class_indices = top1_indices.argmax(dim=1)  # highly likely the self (or another twin in the feature space)
                indices[range(batchsize), max_class_indices, :-1] = indices[range(batchsize), max_class_indices, 1:]
                indices = indices[:, :, :-1]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)
            '''


            ''' M15 noisy kNN; define a buffered pool and select k among them
            topk_sim, indices = classwise_sim.topk(k=self.dm.max_num_samples // 4, dim=-1, largest=True, sorted=True)
            rand_indices = torch.randint(low=0, high=self.dm.max_num_samples // 4, size=[batchsize, self.num_classes, self.args.k])
            rand_indices = rand_indices.to(indices.device)
            indices = torch.gather(input=indices, dim=-1, index=rand_indices)
            '''

            # 1, C, 1
            class_idx = torch.arange(self.num_classes).unsqueeze(0).unsqueeze(-1)
            # C, N, D [[1, C, 1], [B, C, K]] -> B, C, K, D
            knnemb = self.memory_list[class_idx, indices]


            ''' M16 convex-combinated kNN
            cvx_coeff = torch.randn(1, 1, self.args.k, 1).to(knnemb.device).type(knnemb.dtype)
            cvx_coeff = torch.softmax(cvx_coeff, dim=2)
            knnemb = torch.sum(cvx_coeff * knnemb, dim=2)
            knnemb = F.normalize(knnemb, p=2, dim=-1)
            knnemb = torch.cat([repeat(self.global_proto, 'c d -> b c d', b=batchsize), knnemb], dim=1)
            '''


            ''' M17 fixed random pairs
            class_idx = torch.arange(self.num_classes).unsqueeze(0).unsqueeze(-1)

            pairs = []
            for i in idx:
                pair = self.memory_list[:, i + 1:i+self.args.k + 1]
                pairs.append(pair)

            pairs = torch.stack(pairs, dim=0).to(out.device).type(out.dtype)
            knnemb = pairs
            '''

            knnemb = torch.cat([repeat(self.global_proto, 'c d -> b c 1 d', b=batchsize), knnemb], dim=2)
            knnemb = rearrange(knnemb, 'b c k d -> (b c) k d')

        # (B, C, D), (B, CK, D) -> B, C, D
        updated_proto = self.knnformer(repeat(self.global_proto, 'c d -> (repeat c) 1 d', repeat=batchsize), knnemb, knnemb)
        updated_proto = F.normalize(updated_proto, p=2, eps=1e-6, dim=-1)
        sim = torch.einsum('b d, b c d -> b c', out, rearrange(updated_proto, '(b c) 1 d -> b c d', c=self.num_classes))

        return F.log_softmax(sim / 0.01, dim=1)

    def forward_m8_1_vis_and_analy(self, x):
        '''
        <m8.1>
        parellel transformers with global (class-agnostic) kNN

        - parallel input update (knnformer1, knnformer2)
        - avg probs
        - no l2normalization

        -> m24, b5, vis and analyses
        '''

        out = self.backbone(x)
        batchsize = out.shape[0]

        with torch.no_grad():
            if len(self.memory_list.shape) == 3:
                self.memory_list = rearrange(self.memory_list, 'c n d -> (c n) d')

            classwise_sim = torch.einsum('b d, n d -> b n', out, self.memory_list)
            if self.training:  # to ignore self-voting
                # B, C, N -> B, C, K
                topk_sim, indices = classwise_sim.topk(k=self.args.k + 1, dim=-1, largest=True, sorted=True)
                indices = indices[:, 1:]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)

            # N, D [[B, K] -> B, K, D
            knnemb = self.memory_list[indices]

            # B, 1, D
            tr_q = out.unsqueeze(1)
            # (B, 1, D), (B, K, D) -> B, (1 + K), D
            tr_knn_cat = torch.cat([tr_q, knnemb], dim=1)

        tr_q = out.unsqueeze(1)
        qout, _ = self.knnformer(tr_q, tr_q, tr_q)
        nout, attn = self.knnformer2(tr_q, tr_knn_cat, tr_knn_cat)

        classindices = indices // self.dm.max_num_samples
        correct = classindices == y.unsqueeze(1)
        incorrect = classindices != y.unsqueeze(1)
        # knnattn = attn[:, 0, 1:]

        for b in range(256 // 10):
            knnattn = attn[10 * b:10 * (b+1), 0, 1:]

            # begin vis
            import matplotlib.pyplot as plt
            def highlight_cell(x,y, ax=None, **kwargs):
                rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
                ax = ax or plt.gca()
                ax.add_patch(rect)
                return rect

            img = knnattn.detach().cpu().numpy()
            plt.imshow(img)

            for x in range(correct.shape[0]):
                for y in range(correct.shape[1]):
                    if correct[x][y]:
                        highlight_cell(y, x, color="magenta", linewidth=1)

            import pdb ; pdb.set_trace()
        # end vis

        correctknnattn = torch.sum(knnattn * correct, dim=-1) / (torch.sum(correct, dim=-1) + 1e-6)
        incorrectknnattn = torch.sum(knnattn * incorrect, dim=-1) / (torch.sum(incorrect, dim=-1) + 1e-6)

        # b5
        # nout = self.knnformer2(tr_q, tr_q, tr_q)

        # below for m24
        '''
        out = torch.cat([qout[:, 0], nout[:, 0]], dim=1)
        sim = torch.einsum('b d, c d -> b c', out, torch.cat([self.global_proto, self.global_proto], dim=1))

        return F.log_softmax(sim, dim=1)
        '''

        # below for m8.1
        qout = torch.einsum('b d, c d -> b c', qout[:, 0], self.global_proto)
        nout = torch.einsum('b d, c d -> b c', nout[:, 0], self.global_proto)

        avgprob = 0.5 * (F.softmax(qout, dim=1) + F.softmax(nout, dim=1))
        avgprob = torch.clamp(avgprob, 1e-6)  # to prevent numerical unstability

        # k_score = indices.squeeze(1) // self.dm.max_num_samples == y
        k_score = torch.stack(list(map(majority_vote, torch.div(indices, self.dm.max_num_samples, rounding_mode='trunc')))).argmax(dim=1) == y
        correct_before = torch.einsum('b d, c d -> b c', out, self.global_proto).argmax(dim=1) == y
        correct_after = avgprob.argmax(dim=1) == y

        self.k_score += k_score.tolist()
        self.correct_before += correct_before.tolist()
        self.correct_after += correct_after.tolist()
        self.correctknnattn += correctknnattn.tolist()
        self.incorrectknnattn += incorrectknnattn.tolist()

        return torch.log(avgprob)

    def forward_m23(self, x):
        """
        0301 m23

        - generic tokens used
        - parerrel transformer
            - knn1, knn2: prototype update with generic token
            - knn3: prototype self-udpate
        - avged probs
        - l2normalized features used
        """

        out = self.backbone(x)
        # normalize
        out = F.normalize(out, p=2, dim=-1)
        batchsize = out.shape[0]

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, n d -> b n', out, self.memory_list)
            if self.training:  # to ignore self-voting
                # B, N -> B, K
                topk_sim, indices = classwise_sim.topk(k=self.args.k + 1, dim=-1, largest=True, sorted=True)
                indices = indices[:, 1:]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)

            # N, D [[B, K]] -> B, K, D
            knnemb = self.memory_list[indices]

            knnemb = torch.cat([out.unsqueeze(1), knnemb], dim=1)

        # (M, D), (B, CK, D) -> B, M, D
        updated_tokens = self.knnformer(repeat(self.generic_tokens, 'm d -> b m d', b=batchsize), knnemb, knnemb)
        updated_tokens = F.normalize(updated_tokens, p=2, eps=1e-6, dim=-1)

        # (B, (1 + C), D), (B, M, D) -> B, (1 + C), D
        _updated_proto = self.knnformer2(repeat(self.global_proto, 'c d -> b c d', b=batchsize), updated_tokens, updated_tokens)

        # standization & l2 normalization
        # _updated_proto = self.standardize(_updated_proto)

        # output becomes NaN if it's commented!!
        updated_proto = F.normalize(_updated_proto, p=2, eps=1e-6, dim=-1)
        # updated_proto = self.standardize(updated_proto)

        sim = torch.einsum('b d, b c d -> b c', out, updated_proto)

        proto3 = self.knnformer3(self.global_proto.unsqueeze(1), self.global_proto.unsqueeze(1), self.global_proto.unsqueeze(1))

        proto3 = F.normalize(proto3, p=2, eps=1e-6, dim=-1)
        # C, 1, D
        # proto3  = self.standardize(proto3, dim=0)

        sim3 = torch.einsum('b d, c d -> b c', out, proto3.squeeze(1))

        avgprob = 0.5 * (F.softmax(sim / 0.01, dim=1) + F.softmax(sim3 / 0.01, dim=1))
        avgprob = torch.clamp(avgprob, 1e-6)  # to prevent numerical unstability
        return torch.log(avgprob)

    def forward_m192021(self, x):
        """
        m19, m20, m21: the difference between m19~22 comes from concatinating input

        - unnormalized generic tokens as input
        - l2normalized features used
        - sequential update
            1) generic tokens <- kNNs
            2) prototype <- generic tokens
        """

        out = self.backbone(x)
        # normalize
        out = F.normalize(out, p=2, dim=-1)
        batchsize = out.shape[0]

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, n d -> b n', out, self.memory_list)
            if self.training:  # to ignore self-voting
                # B, N -> B, K
                topk_sim, indices = classwise_sim.topk(k=self.args.k + 1, dim=-1, largest=True, sorted=True)
                indices = indices[:, 1:]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)

            # N, D [[B, K]] -> B, K, D
            knnemb = self.memory_list[indices]

            knnemb = torch.cat([out.unsqueeze(1), knnemb], dim=1)

        # (M, D), (B, CK, D) -> B, M, D
        updated_tokens = self.knnformer(repeat(self.generic_tokens, 'm d -> b m d', b=batchsize), knnemb, knnemb)
        updated_tokens = F.normalize(updated_tokens, p=2, eps=1e-6, dim=-1)

        # (B, (1 + C), D), (B, M, D) -> B, (1 + C), D
        feat = torch.cat([out.unsqueeze(1), repeat(self.global_proto, 'c d -> b c d', b=batchsize)], dim=1)
        feat = self.knnformer2(feat, updated_tokens, updated_tokens)

        # standization & l2 normalization
        # _updated_proto = self.standardize(_updated_proto)

        # output becomes NaN if it's commented!!
        feat = F.normalize(feat, p=2, eps=1e-6, dim=-1)

        feat_input, feat_proto = feat[:, 0], feat[:, 1:]
        feat_proto = self.standardize(feat_proto)

        sim = torch.einsum('b d, b c d -> b c', feat_input, feat_proto)
        return F.log_softmax(sim / 0.01, dim=1)

    def forward_b4_linear(self, x):
        '''
        b4
        - concat [input|aknn]
        - pass the tensor to linear
        '''
        out = self.backbone(x)
        batchsize = out.shape[0]

        with torch.no_grad():
            classwise_sim = torch.einsum('b d, n d -> b n', out, self.memory_list)
            if self.training:  # to ignore self-voting
                # B, C, N -> B, C, K
                topk_sim, indices = classwise_sim.topk(k=self.args.k + 1, dim=-1, largest=True, sorted=True)
                indices = indices[:, 1:]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.args.k, dim=-1, largest=True, sorted=True)

            knnemb = self.memory_list[indices]

            # B, 1, D
            tr_q = out.unsqueeze(1)
            # (B, 1, D), (B, C, D) -> B, (1 + C), D
            tr_knn_cat = torch.cat([tr_q, knnemb], dim=1)

        tr_knn_input = tr_knn_cat.view(batchsize, -1)
        updated_input = self.linear(tr_knn_input)

        sim = torch.einsum('b d, c d -> b c', updated_input, self.global_proto)
        return F.log_softmax(sim, dim=1)

    def forward_b6_simplelinear(self, x):
        '''
        b6
        - pass the tensor to linear
        '''
        out = self.backbone(x)
        out = self.linear(out)
        sim = torch.einsum('b d, c d -> b c', out, self.global_proto)

        return F.log_softmax(sim, dim=1)
