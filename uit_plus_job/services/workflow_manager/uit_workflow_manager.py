import os
from jinja2 import Template
from tethysext.atcore.services.workflow_manager.base_workflow_manager import BaseWorkflowManager

from uit_plus_job.models import UitPlusJob


class ResourceWorkflowUitJobManager(BaseWorkflowManager):
    """
    Helper class that prepares and submits uit jobs for resource workflows.
    """

    def prepare(self):
        """
        Prepares job for processing upload to database.

        Returns:
            int: the job id.
        """
        # Prep
        project_id = self.app.get_custom_setting('project_id')
        job_manager = self.app.get_job_manager()

        template_string = """
        #!/usr/bin/bash
        python {{ job_script }} {{ args }}
        """

        script_template = Template(template_string)
        prepared_job_script = script_template.render(job_script=self.job_script, args=self.job_args)

        with open(os.path.join(self.workspace, 'prepared_job_script.sh'), 'w+') as fh:
            fh.write(prepared_job_script)

        # Create Job
        job_kwargs = dict(
            name=self.safe_job_name,
            user=self.user,
            job_type=UitPlusJob,
            project_id=project_id,
            system='onyx',
            node_type='compute',
            num_nodes=1,
            processes_per_node=1,
            queue='debug',
            job_script=prepared_job_script,  # Comes in as the Python file they want to run, but we need to replace with wrapper script
            workspace=self.workspace,
            extended_properties={
                'resource_id': self.resource_id,
                'resource_workflow_id': self.resource_workflow_id,
                'resource_workflow_step_id': self.resource_workflow_step_id,
            }
        )

        job_kwargs.update(self.jobs[0])

        # Generate wrapper script with args as context
        # Override job_script with wrapper script
        # Add (original) job script as input file

        self.workflow = job_manager.create_job(
            **job_kwargs
        )

        # Save prepared job
        self.workflow.save()

        # Save UitJob
        self.prepared = True

        return self.prepared_job.id
