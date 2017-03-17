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

MLRegressionRandomForest <- function(dataset = NULL, options, perform = "run",
									 callback = function(...) 0, ...) {
									  # callback = function(...) list(status = "ok"), ...) {

	print(str(options))
	## Read Dataset ## ----
	variables <- unlist(options[["predictors"]])
	target <- options[["target"]]
	if (target == "") # default for empty target
		target <- NULL

	variables.to.read <- c(target, variables)

	if (is.null(dataset)) { # how to handle factors?

		if (perform == "run") {

			dataset <- .readDataSetToEnd(columns = variables.to.read, exclude.na.listwise = NULL)

		} else {

			dataset <- .readDataSetHeader(columns = variables.to.read)

		}

	} else {

		if (!Sys.getenv("RSTUDIO") == "1") 
			dataset <- .vdf(dataset, columns = variables.to.read)
		
	}

	# This problem can occur when reading the dataset with the columns argument: (reference: http://stackoverflow.com/questions/41441665/how-to-fix-a-malformed-factor)
	# vec <- structure(c(1L,2L, 33L), .Label = c("first", "second"), class = "factor")
	# print(vec)
	# levels(vec) <- levels(vec)
	# print(vec)
	
	# This fixes it:
	colIdx <- sapply(dataset, is.factor)
	if (is.logical(colIdx)) {
		
		seqFactor <- which(colIdx)
		for (var in seqFactor) 
			levels(dataset[[var]]) <- levels(dataset[[var]])
		
	}
	print("str(dataset)")
	print(str(dataset))

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
			list(name = "plotVariableImportance",    type = "image"),
			list(name = "plotTreesVsModelError",     type = "image")
		)
	)

	## Do Analysis ## ----
	errorList <- NULL

	if (is.null(toFromState) && !is.null(variables) && !is.null(target)) { # implies old state was unusable

		# check for errors
		anyErrors <- .hasErrors(dataset = dataset, perform = perform, type = c("infinity", "variance"))
		.pprint(anyErrors)
		doUpdate <- base::identical(anyErrors, FALSE)
		.pprint(doUpdate)
		if (doUpdate) { # do analysis

			toFromState <- .MLRFAnalysis(dataset, purpose = "regression", perform = perform,
										 options = options, variables = variables, target = target)

		} else { # show error message

			errorList <- list(errorType = "badData", errorMessage = anyErrors[["message"]])

		}

	} else { # implies results are retrieved from state

		doUpdate <- TRUE

	}

	## Create Output ## ----
	
	if (doUpdate) { # no errors were found
		
		if (options[["tableVariableImportance"]])
			results[["tableVariableImportance"]] <- .MLRFVarImpTb(toFromState = toFromState, variables = variables, perform = perform)
		
		if (options[["plotTreesVsModelError"]])
			results[["plotTreesVsModelError"]] <- .MLRFplotTreesVsModelError(toFromState = toFromState, options = options,
																					 perform = perform)
		
		if (options[["plotVariableImportance"]])
			results[["plotVariableImportance"]] <- .MLRFplotVariableImportance(toFromState = toFromState, options = options, 
																			   variables = variables, perform = perform)

	} else { # add error messages
		
		# Create an empty table to show the error
		results[["tableVariableImportance"]] <- .MLRFVarImpTb(toFromState = NULL, variables = variables, perform = perform)
		results[["tableVariableImportance"]][["error"]] <- errorList

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
	print(y)
	print(x)
	#print(sprintf("%s = {%s}", y, capture.output(dput(x))))
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

.MLRFplotTreesVsModelError <- function(toFromState, options, perform) {

	rfPlot <- list(
		title = "Number of trees vs model error",
		width = options[["plotWidth"]],
		height = options[["plotHeight"]],
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
		
		rfPlot[["title"]] <- paste0(rfPlot[["title"]], " (", yl, ")") 

		if (res$type != "unsupervised") {
			
			# install.packages("ggplot2", dependencies = TRUE, 
			# 				 lib = "C:/Users/donvd/OneDrive/Documenten/EJgit/build-JASP-JASP_64-Debug/R/library")
			
			g <- drawCanvas(xName = "Trees", yName = yl, xBreaks = pretty(x), yBreaks = pretty(y))
			g <- drawLines(g, df = data.frame(x = x, y = y), color = "gray",  show.legend = FALSE)
			g <- themeJasp(g)

			content <- ""
			if (!Sys.getenv("RSTUDIO") == "1")
				image <- .beginSaveImage(width = options[["plotWidth"]], height = options[["plotHeight"]])
			
			print(g)
			
			if (!Sys.getenv("RSTUDIO") == "1") 
				content <- .endSaveImage(image)
			
			rfPlot[["data"]] <- content
			
		}
		
		rfPlot[["status"]] <- "completed"
		# staterfPlot = rfPlot
		
	}
	
	return(rfPlot)

}

.MLRFplotVariableImportance <- function(toFromState, options, variables, perform) {
	
	rfPlot <- list(
		title = "Relative Importance of Variables",
		width = options[["plotWidth"]],
		height = options[["plotHeight"]],
		custom = list(width = "plotWidth", height = "plotHeight"),
		data = ""
	)
	
	if (perform == "run" && !is.null(toFromState)) { # are there results to plot?
		
		res <- toFromState[["res"]]
		toPlot <- data.frame(
			Feature = variables,
			Importance = unname(randomForest::importance(toFromState$res, type = 1))
		)
		toPlot <- toPlot[order(toPlot[["Importance"]], decreasing = TRUE), ]
		
		axisLimits <- range(pretty(toPlot[["Importance"]]))
		axisLimits[1] <- min(c(0, axisLimits[1]))
		axisLimits[2] <- max(c(0, axisLimits[2]))

		p <- ggplot2::ggplot(toPlot, ggplot2::aes(x=reorder(Feature, Importance), y=Importance)) +
			ggplot2::geom_bar(stat="identity", fill="gray40") +
			ggplot2::coord_flip() + 
			ggplot2::theme_light(base_size=20) +
			ggplot2::scale_x_discrete(name = "") +
			ggplot2::scale_y_continuous(name = "Variable Importance", 
										limits = axisLimits) + 
			# ggplot2::ggtitle("Relative Importance of Variables") +
			ggplot2::theme(plot.title = ggplot2::element_text(size=18))

		content = ""
		if (!Sys.getenv("RSTUDIO") == "1")
			image <- .beginSaveImage(width = options[["plotWidth"]], height = options[["plotHeight"]])
		
		print(p)
		
		if (!Sys.getenv("RSTUDIO") == "1") {
			content <- .endSaveImage(image)
		}
		
		rfPlot[["data"]] <- content
		rfPlot[["status"]] <- "completed"
		
	}
	
	return(rfPlot)
	
}

# plot functions
themeJasp = function(graph, fontsize = 18, legend.position = "auto", xyMargin = "auto",
					 legend.cex = 1.25, axis.title.cex = 1.5, family = NULL,
					 legend.position.left = c(.2, .9), legend.position.right = c(.8, .9)) {

	# "auto" always means: base this on the graph object.
	# In documentation "otherwise" refers to non "auto" usage.

	# graph: graph object to add theme to
	# fontsize: general size of text
	# legend.position: otherwise as described in ?ggplot2::theme
	# xyMargin: otherwise a list where the 1st element is the xMargin and the second the yMargin. Both a numeric vector of length 4
	# legend.cex: legend text has size fontsize*legend.cex
	# axis.title.cex = axis label has size fontsize*axis.title.cex
	# family: font family
	# legend.position.left/ legend.position.right: where to place the legend? (units relative to plot window; c(.5, .5) = center)

	gBuild = ggplot2::ggplot_build(graph)

	# TRUE if graph contains data, FALSE otherwise
	hasData = length(gBuild$data)  != 2

	if (xyMargin == "auto") {

		# get size of axis tick labels
		xBreaks = gBuild$layout$panel_ranges[[1]]$x.labels
		yBreaks = gBuild$layout$panel_ranges[[1]]$y.labels
		xCex = 0# max(nchar(xBreaks, type = "width")) - 4 # is this even necessary ?
		yCex = max(nchar(yBreaks, type = "width")) - 4
		xMargin = c(20 + 5 * xCex, 0, 0, 0) # margin x-label to x-axis
		yMargin = c(0, 20 + 5 * yCex, 0, 0) # margin y-label to y-axis

	} else {

		xMargin = xyMargin[[1]]
		yMargin = xyMargin[[2]]

	}

	# determine legend position
	if (hasData && legend.position == "auto") {

		idxYmax = which.max(gBuild$data[[3]]$y)[1]
		xAtYmax = gBuild$data[[3]]$x[idxYmax]
		xCenter = mean(gBuild$layout$panel_ranges[[1]]$x.range, na.rm = TRUE)
		
		print(gBuild$layout$panel_ranges[[1]]$x.range)
		print(xCenter)

		if (isTRUE(xAtYmax > xCenter)) { # mode right of middle

			legend.position = c(.15, .9)

		} else { # mode left of middle

			legend.position = c(.85, .9)

		}
	}

	return(
		graph + themeJaspRaw(legend.position = legend.position, xMargin = xMargin, yMargin = yMargin,
							 legend.cex = legend.cex, axis.title.cex = axis.title.cex, family = family,
							 fontsize = fontsize)
	)

}

# for manual usage
themeJaspRaw = function(legend.position = "none", xMargin = c(0, 0, 0, 0), yMargin = c(0, 0, 0, 0),
						legend.cex = 1, axis.title.cex = 1, family = NULL, fontsize = 12) {

	ggplot2::theme(
		panel.spacing = grid::unit(2, "cm"),
		panel.grid = ggplot2::element_blank(),
		panel.background = ggplot2::element_rect(color = "white", fill = "white"),
		legend.background = ggplot2::element_rect(color = "white", fill = "white"),
		legend.key = ggplot2::element_rect(color = "white", fill = "white"),
		legend.key.size = grid::unit(2, "cm"),
		legend.text = ggplot2::element_text(family = family, size = legend.cex*fontsize),
		legend.title = ggplot2::element_blank(),
		legend.position = legend.position,
		axis.line = ggplot2::element_blank(),
		axis.ticks = ggplot2::element_line(size = 1.25), # tick width
		axis.ticks.length = grid::unit(.5, "cm"), # tick length
		axis.text = ggplot2::element_text(family = family, size = fontsize),
		axis.title = ggplot2::element_text(family = family, size = axis.title.cex*fontsize),
		# NOT SUPPORTED BY JASPS ggplot 1.0.0 (2.2.1 is the latest version)
		axis.title.x = ggplot2::element_text(margin = ggplot2::margin(xMargin)),
		axis.title.y = ggplot2::element_text(margin = ggplot2::margin(yMargin))
	)

}

drawCanvas = function(xName, yName, xBreaks, yBreaks) {

	graph = ggplot2::ggplot() +
		ggplot2::scale_x_continuous(name = xName, breaks = xBreaks) +
		ggplot2::scale_y_continuous(name = yName, breaks = yBreaks)
	graph = ggplotBtyN(graph, xBreaks = xBreaks, yBreaks = yBreaks, size = 2)

	return(graph)

}

drawLines = function(graph, df, mapping = NULL, size = 1.25, show.legend = TRUE, ...) {

	if (is.null(mapping)) {

		nms = colnames(df)
		
		mapping <- switch(as.character(length(nms)),
			"2" = ggplot2::aes_string(x = nms[1], y = nms[2]),
			"3" = ggplot2::aes_string(x = nms[1], y = nms[2], group = nms[3], linetype = nms[3])
		)		 
		
		# mapping = ggplot2::aes_string(x = nms[1], y = nms[2], group = nms[3], linetype = nms[3])

	}

	graph = graph +
		ggplot2::geom_line(data = df, mapping = mapping, size = size, show.legend = show.legend, ...)

	return(graph)

}

ggplotBtyN = function(graph, xBreaks = NULL, yBreaks = NULL, ...) {

	if (is.null(xBreaks)) {
		# todo get breaks from graph -- is this really necessary?
	}
	if (is.null(yBreaks)) {
		# todo get breaks from graph -- is this really necessary?
	}

	# get x-lim/ y-lim and add 5% of the total range
	xL = c(xBreaks[1], xBreaks[length(xBreaks)]) #+ c(-1, 1) * .02*diff(range(xBreaks))
	yL = c(yBreaks[1], yBreaks[length(yBreaks)]) #+ c(-1, 1) * .01 # *diff(range(xBreaks))

	return(
		graph +
			ggplot2::annotate(geom = "segment", y = -Inf, yend = -Inf,
							  x = xL[1], xend = xL[2], ...) +
			ggplot2::annotate(geom = "segment", x = -Inf, xend = -Inf,
							  y = yL[1], yend = yL[2], ...)
		
	)

}
