import logging
import argparse
from typing import Optional

from kfp import compiler, dsl

import conf
import google.cloud.aiplatform as aip


def delete_old_schedules(schedule_name):
    for schedule in aip.PipelineJobSchedule.list(
        filter=f'display_name="{schedule_name}"', order_by="create_time desc"
    ):
        logging.info(f"Deleting the old schedule {schedule}")
        schedule.delete()


def deploy_pipeline(
    run_type: str,
    pipeline_dsl: str,
    pipeline_root: str,
    pipeline_name: str,
    pipeline_display_name: str,
    pipeline_schedule_name: str,
    gcp_project: str,
    gcp_location: str,
    gcp_service_account: str,
    gcp_network: str,
    gcp_labels: Optional[dict[str, str]] = {},
    cron_string: Optional[str] = "*/15 * * * *",
    template_path: Optional[str] = "pipeline.yaml",
    pipeline_job_parameters: Optional[dict] = {},
    max_concurrent_run_count: Optional[int] = 1,
):
    """
    :param run_type: "single-run" or "scheduled"
    :param pipeline_dsl: pipeline DSL object
    :param pipeline_root: a root folder for the pipelines to keep their intermediate data
    :param pipeline_name: pipeline name
    :param pipeline_display_name: human readable name of the pipeline
    :param cron_string: a string that defines how often the pipeline should run
    :param template_path: a path to a template yaml file (pipeline.yaml by default)
    :param gcp_labels: additional labels to add to the pipeline
    :param pipeline_job_parameters: additional parameters to pass to the pipeline.
        Make sure to set the correct signatures of the pipeline function
    :param max_concurrent_run_count: maximum number of concurrent runs
    """
    if run_type == "scheduled":
        is_scheduled_run = True
    elif run_type == "single-run":
        is_scheduled_run = False
    else:
        raise Exception(f"Unknown run type {run_type}")

    print("### Deployment Summary")
    compiler.Compiler().compile(pipeline_dsl, package_path=template_path)

    aip.init(
        project=gcp_project,
        location=gcp_location,
        service_account=gcp_service_account,
    )

    labels = {"run_type": run_type, "service": pipeline_name}
    labels.update(gcp_labels)

    # Prepare the pipeline job
    job = aip.PipelineJob(
        enable_caching=False,
        display_name=pipeline_display_name,
        template_path=template_path,
        pipeline_root=pipeline_root,
        location=gcp_location,
        labels=labels,
        parameter_values=pipeline_job_parameters,
    )
    if is_scheduled_run:
        delete_old_schedules(pipeline_schedule_name)

        pipeline_job_schedule = aip.PipelineJobSchedule(
            pipeline_job=job, display_name=pipeline_schedule_name
        )

        pipeline_job_schedule.create(
            cron=cron_string,
            max_concurrent_run_count=max_concurrent_run_count,
            service_account=gcp_service_account,
            network=gcp_network,
        )
    else:
        job.submit(
            service_account=gcp_service_account,
            network=gcp_network,
        )


def validate_pipeline(pipeline_dsl, package_path="pipeline.yaml"):
    compiler.Compiler().compile(pipeline_dsl, package_path=package_path)
