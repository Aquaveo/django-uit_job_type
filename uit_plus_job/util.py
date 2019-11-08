"""
********************************************************************************
* Name: util.py
* Author: nswain
* Created On: November 29, 2018
* Copyright: (c) Aquaveo 2018
********************************************************************************
"""
import os
import json
from string import Template

from django.contrib import messages
from django.shortcuts import render, redirect

from tethys_sdk.gizmos import JobsTable

from tethysext.atcore.models.resource_workflow_steps.spatial_rws import SpatialResourceWorkflowStep
from tethysext.atcore.controllers.resource_workflows.map_workflows import MapWorkflowView
from tethysext.atcore.services.condor_workflow_manager import ResourceWorkflowCondorJobManager


class DeltaTemplate(Template):
    delimiter = '%'


def strfdelta(tdelta, fmt):
    """
    Converts the given duration of delta time into H:M:S format.

    Args:
        tdelta(double): duration in time delta.
        fmt(str): duration type

    Returns:
        str: formatted delta time duration value.
    """
    d = {}
    hours, rem = divmod(tdelta.total_seconds(), 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02}'.format(int(hours))
    d["M"] = '{:02}'.format(int(minutes))
    d["S"] = '{:02}'.format(round(seconds))
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


class SpatialUitJobRWS(SpatialResourceWorkflowStep):
    """
    Workflow step used for reviewing previous step parameters and submitting uit jobs.

    Options:
        job(dict): A dictionary containing the kwargs for a UitPlusJob.
    """  # noqa: #501
    CONTROLLER = 'uit_plus_job.util.SpatialUitJobMWV'
    TYPE = 'spatial_uit_job_workflow_step'

    __mapper_args__ = {
        'polymorphic_identity': TYPE
    }

    @property
    def default_options(self):
        default_options = super().default_options
        default_options.update({
            'job': {},
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


class SpatialUitJobMWV(MapWorkflowView):
    """
    Controller for a map workflow view requiring spatial input (drawing).
    """
    template_name = 'atcore/resource_workflows/spatial_condor_job_mwv.html'
    valid_step_classes = [SpatialUitJobRWS]
    previous_steps_selectable = True

    def process_step_options(self, request, session, context, resource, current_step, previous_step, next_step):
        """
        Hook for processing step options (i.e.: modify map or context based on step options).

        Args:
            request(HttpRequest): The request.
            session(sqlalchemy.orm.Session): Session bound to the steps.
            context(dict): Context object for the map view template.
            resource(Resource): the resource for this request.
            current_step(ResourceWorkflowStep): The current step to be rendered.
            previous_step(ResourceWorkflowStep): The previous step.
            next_step(ResourceWorkflowStep): The next step.
        """
        # Turn off feature selection on model layers
        map_view = context['map_view']
        self.set_feature_selection(map_view=map_view, enabled=False)

        # Can run workflows if not readonly
        can_run_workflows = not self.is_read_only(request, current_step)

        # Save changes to map view and layer groups
        context.update({
            'can_run_workflows': can_run_workflows
        })

        # Note: new layer created by super().process_step_options will have feature selection enabled by default
        super().process_step_options(
            request=request,
            session=session,
            context=context,
            resource=resource,
            current_step=current_step,
            previous_step=previous_step,
            next_step=next_step
        )

    def on_get_step(self, request, session, resource, workflow, current_step, previous_step, next_step,
                    *args, **kwargs):
        """
        Hook that is called at the beginning of the get request for a workflow step, before any other controller logic occurs.
            request(HttpRequest): The request.
            session(sqlalchemy.Session): the session.
            resource(Resource): the resource for this request.
            workflow(ResourceWorkflow): The current workflow.
            current_step(ResourceWorkflowStep): The current step to be rendered.
            previous_step(ResourceWorkflowStep): The previous step.
            next_step(ResourceWorkflowStep): The next step.
        Returns:
            None or HttpResponse: If an HttpResponse is returned, render that instead.
        """  # noqa: E501
        step_status = current_step.get_status()
        if step_status != current_step.STATUS_PENDING:
            return self.render_uit_job_table(request, resource, workflow, current_step, previous_step, next_step)

    def render_uit_job_table(self, request, resource, workflow, current_step, previous_step, next_step):
        """
        Render a condor jobs table showing the status of the current job that is processing.
            request(HttpRequest): The request.
            session(sqlalchemy.Session): the session.
            resource(Resource): the resource for this request.
            workflow(ResourceWorkflow): The current workflow.
            current_step(ResourceWorkflowStep): The current step to be rendered.
        Returns:
            HttpResponse: The condor job table view.
        """
        job_id = current_step.get_attribute('condor_job_id')
        app = self.get_app()
        job_manager = app.get_job_manager()
        step_job = job_manager.get_job(job_id=job_id)

        jobs_table = JobsTable(
            job=step_job,
            column_fields=('description', 'creation_time', ),
            hover=True,
            striped=True,
            condensed=False,
            show_detailed_status=True,
            delete_btn=False,
        )

        # Build step cards
        steps = self.build_step_cards(request, workflow)

        # Get the current app
        step_url_name = self.get_step_url_name(request, workflow)

        # Can run workflows if not readonly
        can_run_workflows = not self.is_read_only(request, current_step)

        # Configure workflow lock display
        lock_display_options = self.build_lock_display_options(request, workflow)

        context = {
            'resource': resource,
            'workflow': workflow,
            'steps': steps,
            'current_step': current_step,
            'next_step': next_step,
            'previous_step': previous_step,
            'step_url_name': step_url_name,
            'next_title': self.next_title,
            'finish_title': self.finish_title,
            'previous_title': self.previous_title,
            'back_url': self.back_url,
            'nav_title': '{}: {}'.format(resource.name, workflow.name),
            'nav_subtitle': workflow.DISPLAY_TYPE_SINGULAR,
            'jobs_table': jobs_table,
            'can_run_workflows': can_run_workflows,
            'lock_display_options': lock_display_options
        }

        return render(request, 'atcore/resource_workflows/spatial_condor_jobs_table.html', context)

    def process_step_data(self, request, session, step, model_db, current_url, previous_url, next_url):
        """
        Hook for processing user input data coming from the map view. Process form data found in request.POST and request.GET parameters and then return a redirect response to one of the given URLs.

        Args:
            request(HttpRequest): The request.
            session(sqlalchemy.orm.Session): Session bound to the steps.
            step(ResourceWorkflowStep): The step to be updated.
            model_db(ModelDatabase): The model database associated with the resource.
            current_url(str): URL to step.
            previous_url(str): URL to the previous step.
            next_url(str): URL to the next step.

        Returns:
            HttpResponse: A Django response.

        Raises:
            ValueError: exceptions that occur due to user error, provide helpful message to help user solve issue.
            RuntimeError: exceptions that require developer attention.
        """  # noqa: E501
        if 'next-submit' in request.POST:
            step.validate()

            status = step.get_status(step.ROOT_STATUS_KEY)

            if status != step.STATUS_COMPLETE:
                if status == step.STATUS_WORKING:
                    working_message = step.options.get(
                        'working_message',
                        'Please wait for the job to finish running before proceeding.'
                    )
                    messages.warning(request, working_message)
                elif status in (step.STATUS_ERROR, step.STATUS_FAILED):
                    error_message = step.options.get(
                        'error_message',
                        'The job did not finish successfully. Please press "Rerun" to try again.'
                    )
                    messages.error(request, error_message)
                else:
                    pending_message = step.options.get(
                        'pending_message',
                        'Please press "Run" to continue.'
                    )
                    messages.info(request, pending_message)

                return redirect(request.path)

        return super().process_step_data(request, session, step, model_db, current_url, previous_url, next_url)

    def run_job(self, request, session, resource, workflow_id, step_id, *args, **kwargs):
        """
        Handle run-job-form requests: prepare and submit the condor job.
        """
        if 'run-submit' not in request.POST and 'rerun-submit' not in request.POST:
            return redirect(request.path)

        # Validate data if going to next step
        step = self.get_step(request, step_id, session)

        if self.is_read_only(request, step):
            messages.warning(request, 'You do not have permission to run this workflow.')
            return redirect(request.path)

        # Get options
        scheduler_name = step.options.get('scheduler', None)
        if not scheduler_name:
            raise RuntimeError('Improperly configured SpatialUitJobRWS: no "scheduler" option supplied.')

        job = step.options.get('job', None)
        if not job:
            raise RuntimeError('Improperly configured SpatialUitJobRWS: no "job" option supplied.')

        # Get managers
        model_db, map_manager = self.get_managers(
            request=request,
            resource=resource
        )

        # Get GeoServer Connection Information
        gs_engine = map_manager.spatial_manager.gs_engine

        # Define the working directory
        app = self.get_app()
        working_directory = self.get_working_directory(request, app)

        # Setup the Condor Workflow
        condor_job_manager = ResourceWorkflowCondorJobManager(
            session=session,
            model_db=model_db,
            resource_workflow_step=step,
            jobs=[job],
            user=request.user,
            working_directory=working_directory,
            app=app,
            scheduler_name=scheduler_name,
            gs_engine=gs_engine,
        )

        # Serialize parameters from all previous steps into json
        serialized_params = self.serialize_parameters(step)

        # Write serialized params to file for transfer
        params_file_path = os.path.join(condor_job_manager.workspace, 'workflow_params.json')
        with open(params_file_path, 'w') as params_file:
            params_file.write(serialized_params)

        # Add parameter file to workflow input files
        condor_job_manager.input_files.append(params_file_path)

        # Prepare the job
        job_id = condor_job_manager.prepare()

        # Submit job
        condor_job_manager.run_job()

        # Update status of the resource workflow step
        step.set_status(step.ROOT_STATUS_KEY, step.STATUS_WORKING)
        step.set_attribute(step.ATTR_STATUS_MESSAGE, None)

        # Save the job id to the step for later reference
        step.set_attribute('condor_job_id', job_id)

        # Allow the step to track statuses on each "sub-job"
        step.set_attribute('condor_job_statuses', [])

        # Reset next steps
        step.workflow.reset_next_steps(step)

        session.commit()

        return redirect(request.path)

    @staticmethod
    def get_working_directory(request, app):
        """
        Derive the working directory for the workflow.

        Args:
             request(HttpRequest): Django request instance.
             app(TethysAppBase): App class or instance.

        Returns:
            str: Path to working directory for the workflow.
        """
        user_workspace = app.get_user_workspace(request.user)
        working_directory = user_workspace.path
        return working_directory

    @staticmethod
    def serialize_parameters(step):
        """
        Serialize parameters from previous steps into a file for sending with the workflow.

        Args:
            step(ResourceWorkflowStep): The current step.

        Returns:
            str: path to the file containing serialized parameters.
        """
        parameters = {}
        previous_steps = step.workflow.get_previous_steps(step)

        for previous_step in previous_steps:
            parameters.update({previous_step.name: previous_step.to_dict()})

        return json.dumps(parameters)


class ResourceWorkflowCondorJobManager(object):
    """
    Helper class that prepares and submits condor workflows/jobs for resource workflows.
    """
    ATCORE_EXECUTABLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'resource_workflows')

    def __init__(self, session, model_db, resource_workflow_step, user, working_directory, app, scheduler_name,
                 jobs=None, input_files=None, gs_engine=None, *args):
        """
        Constructor.

        Args:
            session(sqlalchemy.orm.Session): An SQLAlchemy session bound to the resource workflow.
            model_db(ModelDatabase): ModelDatabase instance bound to model database.
            resource_workflow_step(atcore.models.app_users.ResourceWorkflowStep): Instance of ResourceWorkflowStep. Note: Must have active session (i.e. not closed).
            user(auth.User): The Django user submitting the job.
            working_directory(str): Path to users's workspace.
            app(TethysAppBase): Class or instance of an app.
            scheduler_name(str): Name of the condor scheduler to use.
            jobs(list<CondorWorkflowJobNode or dict>): List of CondorWorkflowJobNodes to run.
            input_files(list<str>): List of paths to files to sends as inputs to every job. Optional.
        """  # noqa: E501
        if not jobs or not all(isinstance(x, (dict, CondorWorkflowJobNode)) for x in jobs):
            raise ValueError('Argument "jobs" is not defined or empty. Must provide at least one CondorWorkflowJobNode '
                             'or equivalent dictionary.')

        # DB url for database containing the resource
        self.resource_db_url = str(session.get_bind().url)

        # DB URL for database containing the model database
        self.model_db_url = model_db.db_url

        # Serialize GeoServer Connection
        self.gs_private_url = ''
        self.gs_public_url = ''
        if gs_engine is not None:
            self.gs_private_url, self.gs_public_url = generate_geoserver_urls(gs_engine)

        # Important IDs
        self.resource_id = str(resource_workflow_step.workflow.resource.id)
        self.resource_name = resource_workflow_step.workflow.resource.name
        self.resource_workflow_id = str(resource_workflow_step.workflow.id)
        self.resource_workflow_name = resource_workflow_step.workflow.name
        self.resource_workflow_type = resource_workflow_step.workflow.DISPLAY_TYPE_SINGULAR
        self.resource_workflow_step_id = str(resource_workflow_step.id)
        self.resource_workflow_step_name = resource_workflow_step.name

        # Job Definition Variables
        self.jobs = jobs
        self.jobs_are_dicts = isinstance(jobs[0], dict)
        self.user = user
        self.working_directory = working_directory
        self.app = app
        self.scheduler_name = scheduler_name
        if input_files is None:
            self.input_files = []
        else:
            self.input_files = input_files
        self.custom_job_args = args

        #: Safe name with only A-Z 0-9
        self.safe_job_name = ''.join(s if s.isalnum() else '_' for s in self.resource_workflow_step_name)

        # Prepare standard arguments for all jobs
        self.job_args = [
            self.resource_db_url,
            self.model_db_url,
            self.resource_id,
            self.resource_workflow_id,
            self.resource_workflow_step_id,
            self.gs_private_url,
            self.gs_public_url,
        ]

        # Add custom args
        self.job_args.extend(self.custom_job_args)

        # State variables
        self.workflow = None
        self.prepared = False
        self.workspace_initialized = False
        self._workspace_path = None

    @property
    def workspace(self):
        """
        Workspace path property.
        Returns:
            str: Path to workspace for this workflow
        """
        if self._workspace_path is None:
            self._workspace_path = os.path.join(
                self.working_directory,
                str(self.resource_workflow_id),
                str(self.resource_workflow_step_id),
                self.safe_job_name
            )

            # Initialize workspace
            if not self.workspace_initialized:
                self._initialize_workspace()

        return self._workspace_path

    def _initialize_workspace(self):
        """
        Create workspace if it doesn't exist.
        """
        # Create job directory if it doesn't exist already
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)

        self.workspace_initialized = True

    def prepare(self):
        """
        Prepares all workflow jobs for processing upload to database.

        Returns:
            int: the job id.
        """
        # Prep
        scheduler = get_scheduler(self.scheduler_name)
        # TODO: Cleanup other jobs associated with this workflow...
        job_manager = self.app.get_job_manager()

        # Create Workflow
        self.workflow = job_manager.create_job(
            name=self.safe_job_name,
            description='{}: {}'.format(self.resource_workflow_type, self.resource_workflow_step_name),
            job_type='CONDORWORKFLOW',
            workspace=self.workspace,
            user=self.user,
            scheduler=scheduler,
            extended_properties={
                'resource_id': self.resource_id,
                'resource_workflow_id': self.resource_workflow_id,
                'resource_workflow_step_id': self.resource_workflow_step_id,
            }
        )

        # Save the workflow
        self.workflow.save()

        # Preprocess jobs if they are dicts
        if self.jobs_are_dicts:
            self.jobs = self._build_job_nodes(self.jobs)

        # Add file names as args
        input_file_names = []
        for input_file in self.input_files:
            input_file_name = os.path.split(input_file)[1]
            input_file_names.append(input_file_name)
            self.job_args.append(input_file_name)

        # Parametrize each job
        for job in self.jobs:
            # Set arguments for each job
            job.set_attribute('arguments', self.job_args)

            # Add input files to transfer input files
            transfer_input_files_str = job.get_attribute('transfer_input_files') or ''
            transfer_input_files = transfer_input_files_str.split(',')

            for input_file_name in input_file_names:
                transfer_input_files.append('../{}'.format(input_file_name))

            job.set_attribute('transfer_input_files', transfer_input_files)

            # Add additional remote input file
            remote_input_files = job.remote_input_files
            remote_input_files.extend(self.input_files)
            job.remote_input_files = remote_input_files

            # Save the job
            job.save()

        # Create update status job
        update_status_job = CondorWorkflowJobNode(
            name='finalize',  # Better for display name
            condorpy_template_name='vanilla_transfer_files',
            remote_input_files=[
                os.path.join(self.ATCORE_EXECUTABLE_DIR, 'update_status.py'),
            ],
            workflow=self.workflow
        )

        update_status_job.set_attribute('executable', 'update_status.py')
        update_status_job.set_attribute('arguments', self.job_args)

        update_status_job.save()

        # Bind update_status job only to terminal nodes in the workflow (jobs without children)
        for job in self.jobs:
            if len(job.children_nodes.select_subclasses()) <= 0:
                update_status_job.add_parent(job)

        self.jobs.append(update_status_job)

        update_status_job.save()

        # Save Condor Workflow Job
        self.prepared = True

        return self.workflow.id

    def _build_job_nodes(self, job_dicts):
        """
        Build CondorWorkflowJobNodes from the job_dicts provided.

        Args:
            job_dicts(list<dicts>): A list of dictionaries, each containing the kwargs for a CondorWorkflowJobNode.

        Returns:
            list<CondorWorkflowJobNodes>: the job nodes.
        """
        from tethys_sdk.jobs import CondorWorkflowJobNode

        jobs = []
        job_map = {}

        # Create all the jobs
        for job_dict in job_dicts:
            # Pop-off keys to be handled separately
            parents = job_dict.pop('parents', [])
            attributes = job_dict.pop('attributes', {})

            job_dict.update({'workflow': self.workflow})

            job = CondorWorkflowJobNode(**job_dict)

            for attribute, value in attributes.items():
                job.set_attribute(attribute, value)

            job.save()
            jobs.append(job)

            # For mapping relationships
            job_map[job.name] = {'job': job, 'parents': parents}

        # Set Parent Relationships
        for job in jobs:
            for parent_name in job_map[job.name]['parents']:
                job.add_parent(job_map[parent_name]['job'])

            job.save()

        return jobs

    def run_job(self):
        """
        Prepares and executes the job.

        Returns:
            str: UUID of the CondorWorkflow.
        """
        # Prepare
        if not self.prepared:
            self.prepare()

        # Execute
        self.workflow.execute()
        return str(self.workflow.id)
