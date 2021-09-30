%% 10/15/19: added support for random basis function

% Fixing settings:
% - one learning rate for positive & negative change
% - unbounded feature weights
% - fit to joint - joey style

function out = SSE_featureLearner_random_BS(params, subDat,  basisSet, returnInfo)

    %%% SOME BASICS
    basisSetSize = length(basisSet);
    global Ss Ts softmax

    %%% FREE PARAMETERS
    LRs = [params(1), params(1)];
    invTemp = params(2);
    nLearningParams = 2;
    initialWeights = params((nLearningParams+1):(nLearningParams+basisSetSize));

    %%% INITIALIZE
    storePredDist = nan(height(subDat),2);
    storeParticipantPredictionByConfidence = nan(height(subDat),1);
    
    %%% RUN
    blocks = unique(subDat.Block)';
    for bi = 1:length(blocks)

        % Get data
        block = blocks(bi);
        blockDat = subDat(subDat.Block==block,...
                    {'Trial','S','T','CorrAns','GivenAns','Confidence'});
        numTrials = height(blockDat);
        
        %%% INITIALIZE
        weights = initialWeights;
        if returnInfo
            playerChoices = zeros(4,4);
            participantPredictions = zeros(4,4);
            participantConfidences = zeros(4,4);
            participantPredictionByConfidences = zeros(4,4);
            modelCoopLikelihoods = 0.5*ones(4,4);
            weightsDevelopment = nan(17,basisSetSize);
            weightsDevelopment(1,:) = weights;
        end

        for ti = 1:numTrials

            %%% GAME
            S = blockDat.S(ti);
            T = blockDat.T(ti);

            %%% PREDICTION
            
            % Load participant's prediction
            participantPrediction = strcmp(blockDat.GivenAns{ti},'coop'); % Written as [0, 1]
            participantConfidence = blockDat.Confidence(ti) / 100;
            if participantPrediction == 1
                participantPredictionByConfidence = 0.5 + 0.5 * participantConfidence;
            else
                participantPredictionByConfidence = 0.5 - 0.5 * participantConfidence;
            end
            storeParticipantPredictionByConfidence((bi-1)*16+ti) = participantPredictionByConfidence;
            
            % Make model predictions
            basisFunctionChoices = nan(basisSetSize,1);
            for bfi = 1:basisSetSize
                basisFunctionChoices(bfi) = 2*basisSet(bfi).grid(Ss==S,Ts==T) - 1; % Written as [-1, 1] for input to softmax and weight update equation
            end
            modelPredictionDistribution = softmax([sum(weights.*basisFunctionChoices), 0], invTemp);
            storePredDist((bi-1)*16+ti,:) = modelPredictionDistribution;

            %%% LEARNING
            
            % Outcome
            outcome = strcmp(blockDat.CorrAns{ti},'coop'); % Written as [0, 1]
            
            % Prediction error
            PE = outcome - modelPredictionDistribution(1);

            % Update
            weightsUpdates = basisFunctionChoices * PE;
            weights = weights + weightsUpdates .* LRs(1 + double(weightsUpdates > 0))';
            
            %%% GIMME SOME INFO ABOUT WHAT HAPPENED
            if returnInfo
                participantPredictions(Ss == S, Ts == T) = 2 * participantPrediction - 1; % Write as -1, 1
                participantConfidences(Ss == S, Ts == T) = participantConfidence;
                participantPredictionByConfidences(Ss == S, Ts == T) = participantPredictionByConfidence;
                modelCoopLikelihoods(Ss == S, Ts == T) = modelPredictionDistribution(1);
                playerChoices(Ss == S, Ts == T) = 2 * outcome - 1; % Write as -1, 1
                weightsDevelopment(ti+1,:) = weights;
            end
            
        end
        if returnInfo
            modelCoopLikelihoodsPerBlock.(sprintf('block_%i',bi)) = modelCoopLikelihoods;
            weightsDevelopmentPerBlock.(sprintf('block_%i',bi)) = weightsDevelopment;
        end
    end

    %% Compute likelihood of data given model
    
    %%% ON COMPUTING MODEL ERROR:
    % Simply take the model's p(coop) and fit the participant's
    % prediction-by-confidence ('joint joey style' metric) to
    % that. This is what I did for results on Oct 3!
    allModelCoopLikelihoods = storePredDist(:,1);
    allParticipantCoopLikelihoods = storeParticipantPredictionByConfidence;
    coopLikelihoodErrors = allParticipantCoopLikelihoods - allModelCoopLikelihoods;
    SSE = sum(coopLikelihoodErrors.^2);
    
    % Store variables
    if returnInfo
        % Basics
        out.SSE = SSE;
        out.storePredDist = storePredDist;
        out.storeParticipantPredictionByConfidence = storeParticipantPredictionByConfidence;
        % p of realized choice
        columns = 2 - strcmp(subDat.GivenAns,'coop'); % Column 1 for coop, col 2 for def
        rows = (1:length(storePredDist))';
        indices = sub2ind(size(storePredDist),rows,columns);
        choiceProbs = storePredDist(indices);
        out.choiceProbs = choiceProbs;
        % NLL of realized choices
        choiceNLL = -sum(log(choiceProbs));
        if choiceNLL == Inf
            out.choiceNLL = -log(realmin);
        end
        % Block-wise info
        out.modelCoopLikelihoodsPerBlock = modelCoopLikelihoodsPerBlock;
        out.weightsDevelopmentPerBlock = weightsDevelopmentPerBlock;
        out.basisSet = basisSet;
    else
        out = SSE;
    end

end