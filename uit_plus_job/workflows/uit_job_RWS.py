from tethysext.atcore.models.resource_workflow_steps.spatial_rws import SpatialResourceWorkflowStep


class SpatialUitJobRWS(SpatialResourceWorkflowStep):
    """
    Workflow step used for reviewing previous step parameters and submitting uit jobs.

    Options:
        job(dict): A dictionary containing the kwargs for a UitPlusJob.
    """  # noqa: #501
    CONTROLLER = 'uit_plus_job.workflows.uit_job_MWV.SpatialUitJobMWV'
    TYPE = 'spatial_uit_job_workflow_step'

    __mapper_args__ = {
        'polymorphic_identity': TYPE
    }

    @property
    def default_options(self):
        default_options = super().default_options
        default_options.update({
            'jobs': [{}],
            'job_script': '',
            'working_message': '',
            'error_message': '',
            'pending_message': ''
        })
        return default_options

    def init_parameters(self, *args, **kwargs):
        """
        Initialize the parameters for this step.

        Returns:
            dict<name:dict<help,value>>: Dictionary of all parameters with their initial value set.
        """
        return {}

    def validate(self):
        """
        Validates parameter values of this step.

        Returns:
            bool: True if data is valid, else Raise exception.

        Raises:
            ValueError
        """
        # Run super validate method first to perform built-in checks (e.g.: Required)
        super().validate()