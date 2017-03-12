#
# Copyright (C) 2017 University of Amsterdam
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

MLRegressionRandomForest <- function (dataset = NULL, options, perform = "run",
									  callback = function(...) list(status = "ok"), ...) {

	print(str(options))
	## Read Dataset ## ----
	variables <- unlist(options[["predictors"]])
	target <- options[["target"]]
	if (target == "") # default for empty target
		target <- NULL

	variables.to.read <- c(target, variables)

	if (is.null(dataset)) { # how to handle factors?

		if (perform == "run") {

			dataset <- .readDataSetToEnd(columns.as.numeric = variables.to.read,
										 columns.as.factor = NULL,
										 exclude.na.listwise = NULL)

		} else {

			dataset <- .readDataSetHeader(columns.as.numeric = variables.to.read,
										  columns.as.factor = NULL)

		}

	} else {

		# dataset <- .vdf(dataset,
		# 				columns.as.numeric = variables.to.read,
		# 				columns.as.factor = variables.to.read)

	}

	# ensures order of variables matches order of columns in dataset (first column is target)
	variables <- variables[match(.unv(colnames(dataset)), variables, nomatch = 0L)]

	## TODO: Retrieve State ## ----

	# state <- .retrieveState()

	toFromState <- NULL

	# if (!is.null(state)) {  # is there state?
	#
	# 	diff <- .diff(options, state$options)  # compare old and new options
	#
	# 	# if nothing important was changed retrieve state
	# 	if (is.list(diff) && diff[['variables']] == FALSE) {
	#
	# 		toFromState <- state[["results"]]
	#
	# 	}
	#
	# }

	## Initialize Results ## ----

	results <- list(
		title = "Random Forest Regression",
		.meta = list(
			list(name = "title",                     type = "title"),
			list(name = "tableVariableImportance",   type = "table"),
			list(name = "plotPredictivePerformance", type = "image")
		)
	)

	## Do Analysis ## ----
	errorList <- NULL

	if (is.null(toFromState) && !is.null(variables)) { # implies old state was unusable

		# check for errors
		anyErrors <- .hasErrors(dataset = dataset, perform = perform, type = c("infinity", "variance"))

		doUpdate <- base::identical(anyErrors, FALSE)

		if (doUpdate) { # do analysis

			toFromState <- .MLRFAnalysis(dataset, purpose = "regression", perform = perform,
										 options = options, variables = variables, target = target)

		} else { # show error message

			errorList <- list(errorType = "badData", errorMessage = anyErrors$message)

		}

	} else { # implies results are retrieved from state

		doUpdate <- TRUE

	}

	results[["VariableImportance"]][["error"]] <- errorList

	## Create Output ## ----
	if (doUpdate) {

		if (options[["tableVariableImportance"]])
			results[["tableVariableImportance"]] <- .MLRFVarImpTb(toFromState, variables, perform = perform)

		if (options[["plotPredictivePerformance"]])
			results[["plotPredictivePerformance"]] <- .MLRFplotPredictivePerformance(toFromState, variables, perform = perform)

	}


	## Save State ##


	## Exit Analysis ##
	if (perform == "init") {

		return(list(results=results, status="inited"))#, state=state))

	} else {

		return(list(results=results, status="complete"))#, state=state))

	}
}

.pprint <- function(x) {
	y <- deparse(substitute(x))
	print(sprintf("%s = {%s}", y, capture.output(dput(x))))
}

.MLRFAnalysis <- function(dataset, purpose, options, variables, target, perform) {

	# early exit on unusable input -- needs modification for unsupervised RF
	if (any(perform != "run", is.null(variables), is.null(target)))
		return(NULL)

	preds <- .v(variables) # predictors
	target <- .v(target) # targets

	# defaults for everything set to "auto"
	if (is.character(options[["noOfTrees"]]))
		options[["noOfTrees"]] <- 500

	if (is.character(options[["noOfPredictors"]]))
		options[["noOfPredictors"]] <- ifelse(purpose == "regression",
											  max(c(floor(length(variables) / 3), 1)),
											  floor(sqrt(p)))

	if (is.character(options[["dataBootstrapModel"]]))
		options[["dataBootstrapModel"]] <- 1

	if (is.character(options[["dataTrainingModel"]]))
		options[["dataTrainingModel"]] <- .8

	if (is.character(options[["maximumTerminalNodeSize"]]))
		options[["maximumTerminalNodeSize"]] <- NULL

	if (is.character(options[["minimumTerminalNodeSize"]]))
		options[["minimumTerminalNodeSize"]] <- 1

	# seed
	if (is.numeric(options[["seed"]]))
		set.seed(options[["seed"]])

	# training and test data
	n <- nrow(dataset)

	if (options[["dataTrainingModel"]] < 1) {

		idxTrain <- sample(1:n, floor(options[["dataTrainingModel"]]*n))
		idxTest <- (1:n)[-idxTrain]

	} else {

		idxTrain <- 1:n
		idxTest <- integer(0L)

	}

	if (purpose %in% c("classification", "regression")) {

		xTrain <- dataset[idxTrain, preds, drop = FALSE]
		yTrain <- dataset[idxTrain, target]
		xTest <- dataset[idxTest, preds, drop = FALSE]
		yTest <- dataset[idxTest, target]

	} else { # unsupervised

		xTrain <- dataset[, preds, drop = FALSE]
		yTrain <- NULL
		xTest <- NULL
		yTest <- NULL

	}

	# run RF
	res <- randomForest::randomForest(
		x = xTrain,
		y = yTrain,
		xtest = xTest,
		ytest = yTest,
		ntree = options[["noOfTrees"]],
		mtry = options[["noOfPredictors"]],
		nodesize = options[["minimumTerminalNodeSize"]],
		maxnodes = options[["maximumTerminalNodeSize"]],
		importance = TRUE, # options[["importance"]], # calc importance between rows. Always calc it, only show on user click.
		proximity = FALSE, # options[["proximity"]], # calc proximity between rows. Always calc it, only show on user click.
		keep.forest = TRUE, # should probably always be TRUE (otherwise partialPlot can't be called)
		na.action = randomForest::na.roughfix
	)

	return(list(res = res,
				data = list(xTrain = xTrain, yTrain = yTrain,
							xTest = xTest, yTest = yTest)))

}

.MLRFVarImpTb <- function(toFromState, variables, perform) {

	table <- list(title = "Variable Importance")

	intNms = c("MDiA", "MDiNI") # internal names
	extNms = c("Mean decrease in accuracy", "Mean decrease in node impurity") # external names

	if (any(perform != "run", is.null(toFromState), is.null(variables))) { # no/ bad input

		toTable <- matrix(".", nrow = 1, ncol = 2,
						  dimnames = list(".", intNms))

	} else { # input that can become an actual table

		# matrix for conversion to markup table
		toTable <- randomForest::importance(toFromState$res)
		toTable <- toTable[order(toTable[, 1], decreasing = TRUE), , drop = FALSE]
		colnames(toTable) <- intNms
		rownames(toTable) <- variables

	}

	# fields = list(list(name="case", title="", type="string", combine=TRUE))
	#
	# for (i in seq_along(intNms)) {
	#
	# 	fields[[i]] <- list(name = intNms[i], name = extNms[i], type = type[i],
	#
	# }

	table[["schema"]] <- list(
		fields = list(list(name="case", title="", type="string", combine=TRUE),
					  list(name = intNms[1], title = extNms[1], type="number", format="sf:4;dp:3"),
					  list(name = intNms[2], title = extNms[2], type="number", format="sf:4;dp:3"))
	)

	table[["data"]] <- .MLRFTables(toTable)

	return(table)

}

.MLRFTables <- function(x) {

	n = nrow(x)
	m = ncol(x)

	fieldNames = c("case", colnames(x))
	rmns = rownames(x)

	emptyRow <- vector("list", length = length(fieldNames))
	names(emptyRow) <- fieldNames

	data <- rep(list(emptyRow), n)

	if (is.numeric(x)) { # implies .clean must be used
		for (i in seq_len(n)) {

			data[[i]] <- c(case = rmns[i], lapply(x[i, ], .clean))

		}
	} else {
		for (i in seq_len(n)) {

			data[[i]] <- c(case = rmns[i], x[i, ])

		}
	}

	return(data)

}

.plotMultipleLines = function(x, y, options, Rdebug, ...) {

	args <- list(x = x, y = y, ...)

	# default plotting arguments
	defArgs <- list(type = 'l', lty = 1, ylab = "", xlab = "",
					col = colorspace::rainbow_hcl(NCOL(y)),
					lwd = 3, las = 1, bty = 'n')

	# lookup and apply defaults
	for (name in names(defArgs)) {
		if (is.null(args[[name]]))
			args[[name]] <- defArgs[[name]]
	}

	# par from clean R compendium
	par(cex.main = 1.5, cex.lab = 1.5, cex.axis = 1.3,
		mar = c(5, 6, 4, 5) + 0.1, mgp = c(3.5, 1, 0),
		font.lab = 2)

	# make plot
	content = ""
	if (!Sys.getenv("RSTUDIO") == "1")
		image <- .beginSaveImage(width = 360, height = 240)

	if (NCOL(args[["y"]]) == 1) { 	# default plot
		do.call(graphics::plot, args)
	} else { 						# matplot
		do.call(graphics::matplot, args)
	}
	
	if (!Sys.getenv("RSTUDIO") == "1") {
		content <- .endSaveImage(image)
	} 
		
	return(content)
}

.MLRFplotPredictivePerformance = function(toFromState, options, perform) {

	rfPlot <- list(
		title = "placeholder",
		width = 360,# options[["plotWidth"]],
		height = 240, #options[["plotHeight"]],
		custom = list(width = "plotWidth", height = "plotHeight"),
		data = ""
	)

	if (perform == "run" && !is.null(toFromState)) { # are there results to plot?

		res <- toFromState[["res"]]
		x <- 1:res[["ntree"]]

		if (res[["type"]] == "regression") {

			y <- res[["mse"]]
			yl <- "Mean Squared Error"

		} else if (res[["type"]] == "classification") {

			y <- res[["err.rate"]]
			yl <- "Error rate"

		} else { # res[["type"]] == "unsupervised"

			# there is no plot for unsupervised in the RF package
			# the help page uses MDSplot, but this is something very different
			# it makes a custom call to stats::cmdscales
			# We could make something? I do not know what is regularly used
			# randomForest::MDSplot(res, data[, "Species"])

		}

		rfPlot[["title"]] <- yl # redundant?


		if (res$type != "unsupervised")
			rfPlot[["data"]] <- .plotMultipleLines(x = x, y = y, options = options,
												   xlab = "Trees", ylab = yl)
		
		rfPlot[["status"]] <- "completed"
		# staterfPlot = rfPlot
		
	}
	
	return(rfPlot)

}

