#include "anovaform.h"
#include "ui_anovaform.h"

#include "column.h"
#include "widgets/listmodelvariablesassigned.h"
#include "widgets/listmodelanovamodelnuisancefactors.h"

AnovaForm::AnovaForm(QWidget *parent) :
	AnalysisForm("AnovaForm", parent),
	ui(new Ui::AnovaForm)
{
	ui->setupUi(this);

	ui->listAvailableFields->setModel(&_availableFields);

	_dependentListModel = new ListModelVariablesAssigned(this);
	_dependentListModel->setVariableTypesSuggested(Column::ColumnTypeScale | Column::ColumnTypeOrdinal);
	_dependentListModel->setSource(&_availableFields);
	ui->dependent->setModel(_dependentListModel);

	_fixedFactorsListModel = new ListModelVariablesAssigned(this);
	_fixedFactorsListModel->setSource(&_availableFields);
	_fixedFactorsListModel->setVariableTypesSuggested(Column::ColumnTypeNominal | Column::ColumnTypeOrdinal);
	ui->fixedFactors->setModel(_fixedFactorsListModel);

	_randomFactorsListModel = new ListModelVariablesAssigned(this);
	_randomFactorsListModel->setSource(&_availableFields);
	_randomFactorsListModel->setVariableTypesSuggested(Column::ColumnTypeNominal | Column::ColumnTypeOrdinal);
	ui->randomFactors->setModel(_randomFactorsListModel);

	_repeatedMeasuresListModel = new ListModelVariablesAssigned(this);
	_repeatedMeasuresListModel->setSource(&_availableFields);
	_repeatedMeasuresListModel->setVariableTypesSuggested(Column::ColumnTypeNominal | Column::ColumnTypeOrdinal);
	ui->repeatedMeasures->setModel(_repeatedMeasuresListModel);

	_wlsWeightsListModel = new ListModelVariablesAssigned(this);
	_wlsWeightsListModel->setSource(&_availableFields);
	_wlsWeightsListModel->setVariableTypesSuggested(Column::ColumnTypeScale);
	ui->wlsWeights->setModel(_wlsWeightsListModel);

	ui->buttonAssignDependent->setSourceAndTarget(ui->listAvailableFields, ui->dependent);
	ui->buttonAssignFixed->setSourceAndTarget(ui->listAvailableFields, ui->fixedFactors);
	ui->buttonAssignRandom->setSourceAndTarget(ui->listAvailableFields, ui->randomFactors);
	ui->buttonAssignWLSWeights->setSourceAndTarget(ui->listAvailableFields, ui->wlsWeights);
	ui->buttonAssignRepeated->setSourceAndTarget(ui->listAvailableFields, ui->repeatedMeasures);

	connect(_dependentListModel, SIGNAL(assignmentsChanged()), this, SLOT(dependentChanged()));
	connect(_fixedFactorsListModel, SIGNAL(assignmentsChanged()), this, SLOT(factorsChanged()));
	connect(_randomFactorsListModel, SIGNAL(assignmentsChanged()), this, SLOT(factorsChanged()));
	connect(_repeatedMeasuresListModel, SIGNAL(assignmentsChanged()), this, SLOT(factorsChanged()));

	_anovaModel = new ListModelAnovaModel(this);
	ui->modelTerms->setModel(_anovaModel);

	_contrastsModel = new TableModelVariablesOptions(this);

	connect(_anovaModel, SIGNAL(termsChanged()), this, SLOT(termsChanged()));

	termsChanged();

	_contrastsModel = new TableModelVariablesOptions();
    ui->contrasts->setModel(_contrastsModel);

	ui->containerModel->hide();
	ui->containerFactors->hide();
	ui->containerOptions->hide();
	ui->containerPostHocTests->hide();

}

AnovaForm::~AnovaForm()
{
	delete ui;
}

void AnovaForm::factorsChanged()
{
	QList<ColumnInfo> factorsAvailable;

	factorsAvailable.append(_fixedFactorsListModel->assigned());
	factorsAvailable.append(_randomFactorsListModel->assigned());
	factorsAvailable.append(_repeatedMeasuresListModel->assigned());

	_anovaModel->setVariables(factorsAvailable);
	_contrastsModel->setVariables(factorsAvailable);

	ui->postHocTests_variables->setVariables(factorsAvailable);
}

void AnovaForm::dependentChanged()
{
	const QList<ColumnInfo> &assigned = _dependentListModel->assigned();
	if (assigned.length() == 0)
		_anovaModel->setDependent(ColumnInfo("", 0));
	else
		_anovaModel->setDependent(assigned.last());
}

void AnovaForm::termsChanged()
{
	QList<ColumnInfo> terms = _anovaModel->terms();
	terms.prepend(ColumnInfo("~OVERALL", 0));
	ui->marginalMeans_terms->setVariables(terms);
}
