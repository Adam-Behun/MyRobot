# flow_nodes.py
from pipecat_flows import NodeConfig, FlowResult
import logging

logger = logging.getLogger(__name__)

def create_greeting_node(patient_data: dict) -> NodeConfig:
    """Node 1: Initial greeting and introduction"""
    
    return NodeConfig(
        name="greeting",
        task_messages=[
            {
                "role": "system",
                "content": f"""You are Alexandra from Adam's Medical Practice. You just called an insurance company.

IMPORTANT: You are the CALLER, not the receiver. Wait for the insurance rep to greet you first.

When they answer (e.g., "Hi, this is [name] from [company], how can I help you?"), respond with:
"Hi [their name], this is Alexandra from Adam's Medical Practice. I'm calling to verify eligibility and benefits for a patient."

Keep it brief and natural. Do NOT provide patient details until asked.
"""
            }
        ],
        functions=[],  # No functions needed in greeting
        respond_immediately=False,  # Wait for insurance rep to speak first
        transitions={
            "next": "patient_verification"  # Automatic transition after greeting
        }
    )

def create_patient_verification_node(patient_data: dict) -> NodeConfig:
    """Node 2: Provide patient information when asked"""
    
    # Format DOB properly
    dob = patient_data.get('date_of_birth', 'N/A')
    if dob and dob != 'N/A':
        # Convert "1980-01-01" to "January 1st, 1980" for natural speech
        from datetime import datetime
        try:
            date_obj = datetime.strptime(dob, "%Y-%m-%d")
            dob = date_obj.strftime("%B %d, %Y").replace(" 0", " ")
        except:
            pass
    
    return NodeConfig(
        name="patient_verification",
        task_messages=[
            {
                "role": "system",
                "content": f"""You are verifying a patient. The insurance rep will ask for patient information.

PATIENT INFORMATION TO USE:
- Name: {patient_data.get('patient_name', 'N/A')}
- Date of Birth: {dob}
- Member ID: {patient_data.get('insurance_member_id', 'N/A')}
- CPT Code: {patient_data.get('cpt_code', 'N/A')}
- Provider NPI: {patient_data.get('provider_npi', 'N/A')}

IMPORTANT RULES:
1. Answer their questions directly and naturally
2. If they ask for the patient's name, say: "The patient's name is {patient_data.get('patient_name', 'N/A')}"
3. If they ask for DOB, say: "Date of birth is {dob}"
4. If they ask for member ID, provide it
5. If they ask what procedure/CPT code, say: "We're looking to verify CPT code {patient_data.get('cpt_code', 'N/A')}"
6. Don't volunteer information they haven't asked for yet

Keep responses short and clear. This is a phone conversation."""
            }
        ],
        respond_immediately=False,
        transitions={
            "verified": "authorization_check"  # Move to auth check after verification
        }
    )

def create_authorization_check_node(patient_data: dict) -> NodeConfig:
    """Node 3: Handle authorization status and reference number"""
    
    return NodeConfig(
        name="authorization_check",
        task_messages=[
            {
                "role": "system",
                "content": f"""You are now checking the authorization status for the patient.

IMPORTANT LISTENING POINTS:
1. If they say "approved" or "authorization is approved" → Status is APPROVED
2. If they say "denied" or "not approved" → Status is DENIED  
3. If they say "pending" or "under review" → Status is PENDING
4. If they say "pre-certification needed" or similar → Ask to start one

WHAT TO DO:
- If they put you on hold, say: "Sure, I'll hold" and wait
- If pre-cert is needed, say: "Okay, let's start one"
- When they give you any reference/authorization number, acknowledge it
- ALWAYS ask: "Can I have the reference number for our records?" if not provided

Remember the status and reference number - you'll need them for the database update.

Patient ID for database: {patient_data.get('_id', 'N/A')}"""
            }
        ],
        functions=["update_prior_auth_status"],  # Enable function calling
        respond_immediately=False,
        transitions={
            "complete": "closing"  # Move to closing after getting info
        }
    )

def create_closing_node() -> NodeConfig:
    """Node 4: Thank and close the call"""
    
    return NodeConfig(
        name="closing",
        task_messages=[
            {
                "role": "system",
                "content": """Thank the insurance representative and end the call professionally.

Say something like: "Thank you so much for your help, [their name]. Have a great day!"

Keep it brief and friendly."""
            }
        ],
        respond_immediately=True,  # Respond immediately with thanks
        transitions={}  # No transitions - end of flow
    )

def create_initial_flow(patient_data: dict) -> dict:
    """Create the complete initial flow configuration"""
    
    return {
        "nodes": {
            "greeting": create_greeting_node(patient_data),
            "patient_verification": create_patient_verification_node(patient_data),
            "authorization_check": create_authorization_check_node(patient_data),
            "closing": create_closing_node()
        },
        "initial_node": "greeting",
        "context": {
            "patient_data": patient_data,
            "collected_info": {
                "reference_number": None,
                "auth_status": None,
                "insurance_rep_name": None
            }
        }
    }