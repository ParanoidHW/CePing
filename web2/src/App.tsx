import { useState, useEffect } from 'react'
import { Layout, Typography, Button, Space, message, Steps, Divider, Alert } from 'antd'
import { PlayCircleOutlined, ReloadOutlined, BugOutlined } from '@ant-design/icons'
import {
  WorkloadSelector,
  ModelSelector,
  DynamicForm,
  HardwareForm,
  StrategyForm,
  ResultViewer
} from '@/components'
import { DebugPanel } from '@/components/DebugPanel'
import { useEvaluate } from '@/hooks'
import type { WorkloadSchema, HardwareSchema, StrategySchema, EvaluationRequest } from '@/types'
import '@/utils/requestCapture'

const { Header, Content, Sider } = Layout
const { Title } = Typography

export default function App() {
  const [currentStep, setCurrentStep] = useState(0)
  const [workload, setWorkload] = useState<string | null>(null)
  const [workloadSchema, setWorkloadSchema] = useState<WorkloadSchema | null>(null)
  const [model, setModel] = useState<string | null>(null)
  const [params, setParams] = useState<Record<string, number | string | boolean>>({})
  const [hardware, setHardware] = useState<HardwareSchema>({
    device_preset: '',
    num_devices: 8,
    topology_type: 'mesh'
  })
  const [strategy, setStrategy] = useState<StrategySchema>({
    tp_degree: 1,
    pp_degree: 1,
    dp_degree: 1,
    ep_degree: 1,
    sp_degree: 1,
    activation_checkpointing: false,
    zero_stage: 0
  })
  const [debugPanelVisible, setDebugPanelVisible] = useState(false)

  const { result, loading, error, runEvaluate, reset } = useEvaluate()

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'D') {
        e.preventDefault()
        setDebugPanelVisible(true)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const handleWorkloadNameChange = (name: string) => {
    setWorkload(name)
  }

  const handleWorkloadSchemaReady = (schema: WorkloadSchema) => {
    setWorkloadSchema(schema)
    const defaultParams: Record<string, number | string | boolean> = {}
    Object.entries(schema.parameters).forEach(([key, field]) => {
      defaultParams[key] = field.default ?? ''
    })
    setParams(defaultParams)
    setCurrentStep(1)
  }

  const handleModelChange = (name: string) => {
    setModel(name)
    setCurrentStep(2)
  }

  const handleParamChange = (key: string, value: number | string | boolean) => {
    setParams((prev) => ({ ...prev, [key]: value }))
  }

  const handleEvaluate = async () => {
    if (!workload || !model || !hardware.device_preset) {
      message.error('Please complete all required fields')
      return
    }

    const request: EvaluationRequest = {
      workload_name: workload,
      model_name: model,
      hardware,
      strategy,
      params
    }

    try {
      await runEvaluate(request)
      setCurrentStep(4)
      message.success('Evaluation completed')
    } catch {
      message.error('Evaluation failed')
    }
  }

  const handleReset = () => {
    setCurrentStep(0)
    setWorkload(null)
    setWorkloadSchema(null)
    setModel(null)
    setParams({})
    setHardware({ device_preset: '', num_devices: 8, topology_type: 'mesh' })
    setStrategy({
      tp_degree: 1,
      pp_degree: 1,
      dp_degree: 1,
      ep_degree: 1,
      sp_degree: 1,
      activation_checkpointing: false,
      zero_stage: 0
    })
    reset()
  }

  const steps = [
    { title: 'Workload', description: 'Select workload type' },
    { title: 'Model', description: 'Select model' },
    { title: 'Config', description: 'Configure parameters' },
    { title: 'Evaluate', description: 'Run evaluation' },
    { title: 'Results', description: 'View results' }
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#001529', padding: '0 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={3} style={{ color: '#fff', margin: 0 }}>
          LLM Performance Evaluator
        </Title>
        <Button
          icon={<BugOutlined />}
          onClick={() => setDebugPanelVisible(true)}
          style={{ background: 'transparent', color: '#fff', border: '1px solid #fff' }}
        >
          🔧 Debug
        </Button>
      </Header>
      
      <Layout>
        <Sider width={300} style={{ background: '#fff', padding: '24px' }}>
          <Steps current={currentStep} direction="vertical" items={steps} />
          
          <Divider />
          
          <Space direction="vertical" style={{ width: '100%' }}>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleEvaluate}
              loading={loading}
              disabled={!workload || !model || !hardware.device_preset}
              block
            >
              Run Evaluation
            </Button>
            <Button
              icon={<ReloadOutlined />}
              onClick={handleReset}
              block
            >
              Reset
            </Button>
          </Space>
        </Sider>
        
        <Content style={{ padding: '24px', background: '#f0f2f5' }}>
          {error && (
            <Alert type="error" message="Error" description={error} style={{ marginBottom: 16 }} />
          )}
          
          <WorkloadSelector 
            value={workload} 
            onNameChange={handleWorkloadNameChange}
            onSchemaReady={handleWorkloadSchemaReady}
          />
          
          {workloadSchema && (
            <ModelSelector
              workload={workload}
              value={model}
              onChange={handleModelChange}
            />
          )}
          
          {workloadSchema && workloadSchema.parameters && (
            <DynamicForm
              title="Workload Parameters"
              parameters={workloadSchema.parameters}
              values={params}
              onChange={handleParamChange}
            />
          )}
          
          {model && (
            <>
              <HardwareForm value={hardware} onChange={setHardware} />
              <StrategyForm
                value={strategy}
                onChange={setStrategy}
                maxDevices={hardware.num_devices}
              />
            </>
          )}
          
          <ResultViewer result={result} loading={loading} />
        </Content>
      </Layout>

      <DebugPanel
        visible={debugPanelVisible}
        onClose={() => setDebugPanelVisible(false)}
        workload={workload}
        model={model}
        hardware={hardware}
        strategy={strategy}
        params={params}
      />
    </Layout>
  )
}