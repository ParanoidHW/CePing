import React, { useState } from 'react'
import { Layout, Typography, Button, Space, message, Steps, Divider, Card, Alert } from 'antd'
import { PlayCircleOutlined, ResetOutlined } from '@ant-design/icons'
import {
  WorkloadSelector,
  ModelSelector,
  DynamicForm,
  HardwareForm,
  StrategyForm,
  ResultViewer
} from '@/components'
import { useEvaluate } from '@/hooks'
import type { WorkloadSchema, ModelSchema, HardwareSchema, StrategySchema, EvaluationRequest } from '@/types'

const { Header, Content, Sider } = Layout
const { Title } = Typography

export default function App() {
  const [currentStep, setCurrentStep] = useState(0)
  const [workload, setWorkload] = useState<string | null>(null)
  const [workloadSchema, setWorkloadSchema] = useState<WorkloadSchema | null>(null)
  const [model, setModel] = useState<string | null>(null)
  const [modelSchema, setModelSchema] = useState<ModelSchema | null>(null)
  const [params, setParams] = useState<Record<string, number | string | boolean>>({})
  const [hardware, setHardware] = useState<HardwareSchema>({
    device: '',
    num_devices: 8,
    topology: undefined
  })
  const [strategy, setStrategy] = useState<StrategySchema>({
    tp_degree: 1,
    pp_degree: 1,
    dp_degree: 1,
    sp_degree: 1
  })

  const { result, loading, error, runEvaluate, reset } = useEvaluate()

  const handleWorkloadChange = (name: string, schema: WorkloadSchema) => {
    setWorkload(name)
    setWorkloadSchema(schema)
    const defaultParams: Record<string, number | string | boolean> = {}
    Object.entries(schema.parameters).forEach(([key, field]) => {
      defaultParams[key] = field.default
    })
    setParams(defaultParams)
    setCurrentStep(1)
  }

  const handleModelChange = (name: string, schema: ModelSchema) => {
    setModel(name)
    setModelSchema(schema)
    setCurrentStep(2)
  }

  const handleParamChange = (key: string, value: number | string | boolean) => {
    setParams((prev) => ({ ...prev, [key]: value }))
  }

  const handleEvaluate = async () => {
    if (!workload || !model || !hardware.device) {
      message.error('Please complete all required fields')
      return
    }

    const request: EvaluationRequest = {
      workload: {
        name: workload,
        parameters: params
      },
      model: {
        name: model
      },
      hardware,
      strategy
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
    setModelSchema(null)
    setParams({})
    setHardware({ device: '', num_devices: 8, topology: undefined })
    setStrategy({ tp_degree: 1, pp_degree: 1, dp_degree: 1, sp_degree: 1 })
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
      <Header style={{ background: '#001529', padding: '0 24px' }}>
        <Title level={3} style={{ color: '#fff', margin: '16px 0' }}>
          LLM Performance Evaluator
        </Title>
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
              disabled={!workload || !model || !hardware.device}
              block
            >
              Run Evaluation
            </Button>
            <Button
              icon={<ResetOutlined />}
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
          
          <WorkloadSelector value={workload} onChange={handleWorkloadChange} />
          
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
    </Layout>
  )
}